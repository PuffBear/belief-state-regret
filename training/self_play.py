"""
Self-Play Engine for B-SRM-CHFA Training  — v3
Implements Algorithm 1 from the paper.

v3 fixes:
  - Target network for stable value estimates (Polyak-averaged)
  - Exploration schedule (ε-greedy decaying from 0.5 → 0.05)
  - Multiple value network updates per episode
  - Value experience replay buffer for stable V learning
  - Counterfactual regrets via env save/restore
  - End-to-end encoder+belief training with raw logits
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque

from environments.pursuit_evasion_env import PursuitEvasionEnv, Action, Observation, AgentState
from models.history_encoder import ContextualHistoryEncoder, HistoryBuffer, compute_opponent_movement_label
from models.belief_engine import BeliefInferenceEngine, GoalConditionedValueNetwork
from models.regret_network import NeuralRegretMinimizer
from training.reservoir_buffer import ReservoirBuffer


# ======================================================================
# Helper: save / restore environment state
# ======================================================================

def _save_env_state(env: PursuitEvasionEnv) -> dict:
    return {
        "pursuer_positions": [p.position for p in env.pursuers],
        "evader_positions":  [e.position for e in env.evaders],
        "timestep": env.timestep,
        "done": env.done,
    }

def _restore_env_state(env: PursuitEvasionEnv, snap: dict):
    for p, pos in zip(env.pursuers, snap["pursuer_positions"]):
        p.position = pos
    for e, pos in zip(env.evaders, snap["evader_positions"]):
        e.position = pos
    env.timestep = snap["timestep"]
    env.done = snap["done"]


# ======================================================================
# Value Experience Replay
# ======================================================================

class ValueReplayBuffer:
    """Simple circular buffer for (s, s', cost, done) transitions."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, next_state, cost, done):
        self.buffer.append((
            state.detach().cpu(),
            next_state.detach().cpu(),
            cost,
            done,
        ))

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]

        s  = torch.cat([b[0] for b in batch]).to(device)
        sn = torch.cat([b[1] for b in batch]).to(device)
        c  = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        d  = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        return s, sn, c, d

    def __len__(self):
        return len(self.buffer)


class SelfPlayEngine:
    """
    Runs self-play episodes and computes counterfactual regrets.
    Now with exploration schedule.
    """

    def __init__(
        self,
        env: PursuitEvasionEnv,
        encoder: ContextualHistoryEncoder,
        belief_engine: BeliefInferenceEngine,
        regret_net: NeuralRegretMinimizer,
        value_net: GoalConditionedValueNetwork,
        target_value_net: GoalConditionedValueNetwork,
        reservoir: ReservoirBuffer,
        device: torch.device,
        evader_policy: str = "random",
        epsilon: float = 0.5,
    ):
        self.env = env
        self.encoder = encoder
        self.belief_engine = belief_engine
        self.regret_net = regret_net
        self.value_net = value_net
        self.target_value_net = target_value_net
        self.reservoir = reservoir
        self.device = device
        self.evader_policy = evader_policy
        self.n_actions = 5
        self.epsilon = epsilon  # exploration rate

    # ------------------------------------------------------------------
    # Phase 1: Episode Rollout
    # ------------------------------------------------------------------

    def run_episode(self) -> Dict:
        pursuer_obs, evader_obs = self.env.reset()
        buffers = [HistoryBuffer(max_history_len=self.env.max_steps)
                   for _ in range(self.env.n_pursuers)]

        traj = {
            "embeddings_p": [], "states": [],
            "actions_p": [], "actions_e": [],
            "costs_p": [], "obs_p": [],
            "raw_obs_seqs": [], "env_snapshots": [],
        }

        for t in range(self.env.max_steps):
            traj["env_snapshots"].append(_save_env_state(self.env))
            traj["states"].append(
                torch.from_numpy(self.env.get_state()).float().unsqueeze(0).to(self.device)
            )

            z_ts = []
            pursuer_actions = []
            raw_seqs_this_step = []

            for i in range(self.env.n_pursuers):
                if len(buffers[i]) == 0:
                    z_t = torch.zeros(1, self.encoder.hidden_dim, device=self.device)
                    raw_seqs_this_step.append(None)
                else:
                    obs_seq, act_seq, cost_seq = buffers[i].get_sequence()
                    raw_seqs_this_step.append((obs_seq.copy(), act_seq.copy(), cost_seq.copy()))
                    obs_t  = torch.from_numpy(obs_seq).unsqueeze(0).float().to(self.device)
                    act_t  = torch.from_numpy(act_seq).unsqueeze(0).to(self.device)
                    cost_t = torch.from_numpy(cost_seq).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        z_t, _, _ = self.encoder(obs_t, act_t, cost_t)

                z_ts.append(z_t)

                # ε-greedy on top of regret-matching policy
                if np.random.random() < self.epsilon:
                    action_idx = np.random.randint(0, self.n_actions)
                else:
                    with torch.no_grad():
                        policy = self.regret_net.get_regret_matching_policy(z_t)
                        action_idx = torch.multinomial(policy, 1).item()
                pursuer_actions.append(Action(action_idx))

            evader_actions = self._get_evader_actions(evader_obs)

            traj["embeddings_p"].append(z_ts)
            traj["actions_p"].append(pursuer_actions)
            traj["actions_e"].append(evader_actions)
            traj["obs_p"].append(pursuer_obs)
            traj["raw_obs_seqs"].append(raw_seqs_this_step)

            pursuer_obs, evader_obs, p_costs, e_costs, done = self.env.step(
                pursuer_actions, evader_actions
            )
            traj["costs_p"].append(p_costs)

            for i in range(self.env.n_pursuers):
                curr_det = pursuer_obs[i].detected_opponents
                prev_det = traj["obs_p"][-1][i].detected_opponents if traj["obs_p"] else []
                opp_label = compute_opponent_movement_label(prev_det, curr_det)
                buffers[i].add(pursuer_obs[i], pursuer_actions[i].value, p_costs[i],
                               opponent_movement=opp_label)

            if done:
                traj["states"].append(
                    torch.from_numpy(self.env.get_state()).float().unsqueeze(0).to(self.device)
                )
                break

        traj["done"]      = done
        traj["capture"]   = self.env.captured
        traj["timesteps"] = self.env.timestep
        return traj

    def compute_regrets(self, trajectory: Dict) -> int:
        """
        Compute counterfactual regrets using MONTE CARLO returns as baseline.

        For each timestep t, pursuer p_idx:
          - baseline = actual cost-to-go from t (sum of remaining costs)
          - For each action a: simulate 1 step with env rollback,
            then use (cf_cost_step + remaining_cost_to_go) as counterfactual value
          - regret(a) = baseline - counterfactual(a)
            Positive → action a would have been cheaper → should play more

        This avoids relying on V(s) estimates which were unstable.
        """
        embeddings  = trajectory["embeddings_p"]
        actions_p   = trajectory["actions_p"]
        actions_e   = trajectory["actions_e"]
        costs_p     = trajectory["costs_p"]
        snapshots   = trajectory["env_snapshots"]
        T           = len(embeddings)
        total_added = 0

        # Pre-compute actual cost-to-go for each (timestep, pursuer)
        # cost_to_go[t][p] = sum of costs from step t to end for pursuer p
        cost_to_go = [[0.0] * self.env.n_pursuers for _ in range(T + 1)]
        for t in range(T - 1, -1, -1):
            for p in range(self.env.n_pursuers):
                cost_to_go[t][p] = costs_p[t][p] + cost_to_go[t + 1][p]

        for t in range(T):
            for p_idx in range(self.env.n_pursuers):
                z_t = embeddings[t][p_idx]
                baseline = cost_to_go[t][p_idx]  # actual cost-to-go

                # Remaining cost after this step (for counterfactual calc)
                remaining_after_step = cost_to_go[t + 1][p_idx] if t + 1 < T else 0.0

                for a in range(self.n_actions):
                    # Counterfactual 1-step: what cost if we took action a?
                    _restore_env_state(self.env, snapshots[t])
                    cf_p_actions = list(actions_p[t])
                    cf_p_actions[p_idx] = Action(a)

                    _, _, cf_costs, _, cf_done = self.env.step(cf_p_actions, actions_e[t])

                    if cf_done:
                        cf_value = cf_costs[p_idx]  # terminal
                    else:
                        # 1-step counterfactual cost + rest of trajectory cost
                        cf_value = cf_costs[p_idx] + remaining_after_step

                    # regret(a) = baseline - cf_value
                    # positive = action a was cheaper than what happened
                    regret = baseline - cf_value

                    self.reservoir.add(z_t.squeeze(0), a, regret)
                    total_added += 1

        # Mark env as done so next run_episode() will call reset()
        self.env.done = True

        return total_added

    def _get_evader_actions(self, evader_obs: List[Observation]) -> List[Action]:
        if self.evader_policy == "random":
            return [Action(np.random.randint(0, 5)) for _ in range(self.env.n_evaders)]
        elif self.evader_policy == "flee":
            actions = []
            for obs in evader_obs:
                if obs.detected_opponents:
                    avg_dx = np.mean([d[0] for d in obs.detected_opponents])
                    avg_dy = np.mean([d[1] for d in obs.detected_opponents])
                    if abs(avg_dx) > abs(avg_dy):
                        actions.append(Action.WEST if avg_dx > 0 else Action.EAST)
                    else:
                        actions.append(Action.NORTH if avg_dy > 0 else Action.SOUTH)
                else:
                    actions.append(Action(np.random.randint(0, 5)))
            return actions
        return [Action(np.random.randint(0, 5)) for _ in range(self.env.n_evaders)]


# ======================================================================
# Trainer v3
# ======================================================================

class Trainer:
    """
    Full B-SRM-CHFA training loop.

    v3 improvements:
    - Target network for value function (Polyak τ=0.005)
    - Value experience replay buffer (100K transitions)
    - ε-greedy exploration (0.5 → 0.05 over training)
    - Multiple value updates per episode (n_value_updates=4)
    """

    def __init__(
        self,
        grid_size: int = 10,
        n_pursuers: int = 3,
        n_evaders: int = 2,
        max_steps: int = 100,
        embedding_dim: int = 256,
        reservoir_capacity: int = 1_000_000,
        lr: float = 3e-4,
        batch_size: int = 256,
        update_frequency: int = 5,
        eval_interval: int = 100,
        lambda_opponent: float = 0.1,
        device: str = "cpu",
        evader_policy: str = "random",
        seed: Optional[int] = None,
        # v3 hyperparams
        tau: float = 0.005,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 5000,
        n_value_updates: int = 4,
        value_buffer_capacity: int = 100_000,
    ):
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.eval_interval = eval_interval
        self.lambda_opponent = lambda_opponent
        self.device = torch.device(device)
        self.seed = seed
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.tau = tau
        self.n_value_updates = n_value_updates

        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        # Environment
        self.env = PursuitEvasionEnv(
            grid_size=grid_size, n_pursuers=n_pursuers,
            n_evaders=n_evaders, max_steps=max_steps, seed=seed,
        )

        n_agents = n_pursuers + n_evaders
        state_dim = 2 * n_agents + 1

        # Models
        self.encoder = ContextualHistoryEncoder(
            obs_shape=(5, 5, 3), n_actions=5,
            hidden_dim=embedding_dim, opponent_modeling=True,
        ).to(self.device)

        self.belief_engine = BeliefInferenceEngine(
            embedding_dim=embedding_dim,
            grid_size=grid_size, n_agents=n_agents,
        ).to(self.device)

        self.regret_net = NeuralRegretMinimizer(
            embedding_dim=embedding_dim, n_actions=5,
        ).to(self.device)

        self.value_net = GoalConditionedValueNetwork(
            state_dim=state_dim,
        ).to(self.device)

        # Target network (frozen copy, updated via Polyak averaging)
        self.target_value_net = GoalConditionedValueNetwork(
            state_dim=state_dim,
        ).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        for p in self.target_value_net.parameters():
            p.requires_grad = False

        # Buffers
        self.reservoir = ReservoirBuffer(capacity=reservoir_capacity)
        self.value_buffer = ValueReplayBuffer(capacity=value_buffer_capacity)

        # Self-play engine
        self.engine = SelfPlayEngine(
            env=self.env, encoder=self.encoder,
            belief_engine=self.belief_engine,
            regret_net=self.regret_net,
            value_net=self.value_net,
            target_value_net=self.target_value_net,
            reservoir=self.reservoir,
            device=self.device, evader_policy=evader_policy,
            epsilon=epsilon_start,
        )

        # Optimizers
        self.opt_enc_belief = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.belief_engine.parameters()),
            lr=lr,
        )
        self.opt_regret = torch.optim.Adam(self.regret_net.parameters(), lr=lr)
        self.opt_value  = torch.optim.Adam(self.value_net.parameters(), lr=lr * 3)  # faster for value

        self.metrics_history: List[Dict] = []
        self._print_model_summary()

    def _print_model_summary(self):
        total = 0
        for name, model in [("Encoder", self.encoder),
                            ("Belief Engine", self.belief_engine),
                            ("Regret Network", self.regret_net),
                            ("Value Network", self.value_net)]:
            n = sum(p.numel() for p in model.parameters())
            total += n
            print(f"  {name:20s}: {n:>10,} params")
        print(f"  {'TOTAL':20s}: {total:>10,} params\n")

    def _get_epsilon(self, episode: int) -> float:
        """Linear decay from epsilon_start to epsilon_end."""
        frac = min(1.0, episode / self.epsilon_decay_episodes)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def _soft_update_target(self):
        """Polyak averaging: target ← τ·online + (1-τ)·target"""
        for tp, p in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self, num_episodes: int = 10_000) -> List[Dict]:
        from tqdm import tqdm

        print(f"Training: {num_episodes} ep, {self.grid_size}×{self.grid_size} grid, "
              f"ε={self.epsilon_start}→{self.epsilon_end}, τ={self.tau}")
        print("-" * 60)

        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            # Update exploration rate
            self.engine.epsilon = self._get_epsilon(episode)

            # Phase 1: Self-play
            trajectory = self.engine.run_episode()

            # Store transitions in value buffer
            self._store_value_transitions(trajectory)

            # Phase 2: Compute regrets
            n_samples = self.engine.compute_regrets(trajectory)

            # Phase 3: Network updates
            losses = {}
            if episode % self.update_frequency == 0 and self.reservoir.is_ready:
                losses = self._update_networks(trajectory)

            # Soft-update target network
            self._soft_update_target()

            metrics = {
                "episode": episode,
                "timesteps": trajectory["timesteps"],
                "capture": trajectory["capture"],
                "total_cost": sum(sum(c) for c in trajectory["costs_p"]),
                "reservoir_size": len(self.reservoir),
                "epsilon": self.engine.epsilon,
            }
            metrics.update(losses)
            self.metrics_history.append(metrics)

            if episode % self.eval_interval == 0:
                self._log_progress(episode, num_episodes)

            # Periodically clear old reservoir entries to prevent stale embeddings
            if episode % 2000 == 0 and episode > 0:
                old_size = len(self.reservoir)
                self.reservoir = ReservoirBuffer(capacity=self.reservoir.capacity)
                self.engine.reservoir = self.reservoir
                print(f"  [Reservoir cleared at ep {episode} (was {old_size} entries)]")

        return self.metrics_history

    def _store_value_transitions(self, trajectory: Dict):
        """Add all trajectory transitions to value replay buffer."""
        states = trajectory["states"]
        costs_p = trajectory["costs_p"]
        for t in range(len(costs_p)):
            if t + 1 < len(states):
                is_terminal = 1.0 if (t == len(costs_p) - 1 and trajectory["done"]) else 0.0
                self.value_buffer.add(
                    states[t], states[t + 1],
                    costs_p[t][0],  # first pursuer's cost
                    is_terminal,
                )

    # ------------------------------------------------------------------
    # Network updates
    # ------------------------------------------------------------------

    def _update_networks(self, trajectory: Dict) -> Dict:
        losses = {}

        # ---- 1. Regret network ----
        z_b, a_b, r_b = self.reservoir.sample_batch(self.batch_size, device=self.device)
        pred_r = self.regret_net(z_b, a_b).squeeze(-1)
        loss_regret = F.mse_loss(pred_r, r_b)

        self.opt_regret.zero_grad()
        loss_regret.backward()
        torch.nn.utils.clip_grad_norm_(self.regret_net.parameters(), 1.0)
        self.opt_regret.step()
        losses["loss_regret"] = loss_regret.item()

        # ---- 2. Value network (multiple updates from replay buffer) ----
        if len(self.value_buffer) >= 64:
            val_losses = []
            for _ in range(self.n_value_updates):
                s, sn, c, d = self.value_buffer.sample(self.batch_size, self.device)

                # Use TARGET network for TD targets
                with torch.no_grad():
                    v_next = self.target_value_net(sn) * (1.0 - d)
                td_target = c + v_next

                v_pred = self.value_net(s)
                loss_v = F.mse_loss(v_pred, td_target)

                self.opt_value.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.opt_value.step()
                val_losses.append(loss_v.item())

            losses["loss_value"] = np.mean(val_losses)

        # ---- 3. Encoder + Belief (end-to-end) ----
        states = trajectory["states"]
        raw_seqs = trajectory["raw_obs_seqs"]
        n_agents = self.n_pursuers + self.n_evaders

        usable = [(t, raw_seqs[t][0]) for t in range(len(raw_seqs))
                  if raw_seqs[t][0] is not None]

        if usable:
            n_belief = min(len(usable), 32)
            chosen = [usable[i] for i in np.random.choice(len(usable), n_belief, replace=False)]

            z_list, true_pos_list = [], []
            for t, (obs_np, act_np, cost_np) in chosen:
                obs_t  = torch.from_numpy(obs_np).unsqueeze(0).float().to(self.device)
                act_t  = torch.from_numpy(act_np).unsqueeze(0).to(self.device)
                cost_t = torch.from_numpy(cost_np).unsqueeze(0).to(self.device)

                z_t, _, _ = self.encoder(obs_t, act_t, cost_t)   # WITH grad
                z_list.append(z_t)

                raw_state = states[t].squeeze(0)
                positions = []
                for ag in range(n_agents):
                    x = raw_state[2 * ag].item() * self.grid_size
                    y = raw_state[2 * ag + 1].item() * self.grid_size
                    positions.append([x, y])
                true_pos_list.append(positions)

            z_cat    = torch.cat(z_list, dim=0)
            true_pos = torch.tensor(true_pos_list, dtype=torch.float32, device=self.device)

            pred_logits = self.belief_engine(z_cat, return_logits=True)
            loss_bel    = self.belief_engine.compute_loss(pred_logits, true_pos)

            self.opt_enc_belief.zero_grad()
            loss_bel.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.belief_engine.parameters(), 1.0)
            self.opt_enc_belief.step()
            losses["loss_belief"] = loss_bel.item()

        return losses

    # ------------------------------------------------------------------
    # Logging / Evaluation / Checkpointing
    # ------------------------------------------------------------------

    def _log_progress(self, episode, total):
        window = min(self.eval_interval, len(self.metrics_history))
        recent = self.metrics_history[-window:]
        avg_t  = np.mean([m["timesteps"] for m in recent])
        cap    = np.mean([m["capture"]   for m in recent])
        cost   = np.mean([m["total_cost"] for m in recent])
        eps    = recent[-1].get("epsilon", 0)

        parts = []
        for key in ["loss_regret", "loss_belief", "loss_value"]:
            vals = [m[key] for m in recent if key in m]
            if vals: parts.append(f"{key}={np.mean(vals):.4f}")

        print(f"\n  [Ep {episode}/{total} | ε={eps:.3f} | AvgTime={avg_t:.1f} | Cap={cap:.1%} "
              f"| Cost={cost:.1f} | Res={len(self.reservoir)} | {' | '.join(parts)}]")

    def evaluate(self, n_trials: int = 100) -> Dict:
        self.encoder.eval(); self.regret_net.eval()
        old_eps = self.engine.epsilon
        self.engine.epsilon = 0.0  # no exploration during eval
        res = {"timesteps": [], "captures": [], "costs": []}
        for _ in range(n_trials):
            t = self.engine.run_episode()
            res["timesteps"].append(t["timesteps"])
            res["captures"].append(t["capture"])
            res["costs"].append(sum(sum(c) for c in t["costs_p"]))
        self.engine.epsilon = old_eps
        self.encoder.train(); self.regret_net.train()
        return {
            "avg_timesteps": np.mean(res["timesteps"]),
            "std_timesteps": np.std(res["timesteps"]),
            "capture_rate":  np.mean(res["captures"]),
            "avg_cost":      np.mean(res["costs"]),
        }

    def save_checkpoint(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "belief_engine": self.belief_engine.state_dict(),
            "regret_net": self.regret_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "target_value_net": self.target_value_net.state_dict(),
            "metrics": self.metrics_history,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.belief_engine.load_state_dict(ckpt["belief_engine"])
        self.regret_net.load_state_dict(ckpt["regret_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        if "target_value_net" in ckpt:
            self.target_value_net.load_state_dict(ckpt["target_value_net"])
        if "metrics" in ckpt:
            self.metrics_history = ckpt["metrics"]
        print(f"Checkpoint loaded ← {path}")
