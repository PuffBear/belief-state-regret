"""
Self-Play Training for B-SRM-CHFA — v2 (Two-Phase with CTDE)

Architecture per episode:
  Rollout → Phase 1 (representation) → Phase 2 (regret) → Value update

Phase 1: Encoder + Belief + Opponent training (WITH gradients, using CTDE ground truth)
Phase 2: Regret network training (detached z_t, Deep CFR style)

The two phases are kept strictly separate:
  - Phase 1 uses raw_buffer + encoder/belief/opponent optimizers
  - Phase 2 uses regret reservoir + regret_optimizer
  - No gradient flows from Phase 2 into Phase 1 networks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional
from collections import deque
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environments.pursuit_evasion_env import PursuitEvasionEnv, Action
from agents.bsrmchfa_agent import BSRMCHFAAgent
from training.raw_buffer import RawExperienceBuffer, ValueBuffer


# ============================================================
# Helpers
# ============================================================

def obs_to_dict(observation, grid_size=25):
    """Convert environment Observation dataclass to agent dict format."""
    fov_grid = observation.local_grid.astype(np.float32)
    own_pos = np.array(observation.self_position, dtype=np.float32)
    if observation.detected_opponents:
        dx, dy = observation.detected_opponents[0]
        last_known_opponent = np.array([dx, dy, 0], dtype=np.float32)
    else:
        last_known_opponent = np.array([0, 0, observation.timestep], dtype=np.float32)
    return {
        'fov_grid': fov_grid,
        'own_pos': own_pos,
        'last_known_opponent': last_known_opponent
    }


def save_env_state(env):
    """Save environment state for counterfactual rollback."""
    return {
        'pursuer_positions': [p.position for p in env.pursuers],
        'evader_positions': [e.position for e in env.evaders],
        'timestep': env.timestep,
        'done': env.done,
        'captured': env.captured,
    }


def restore_env_state(env, snap):
    """Restore environment state from snapshot."""
    for p, pos in zip(env.pursuers, snap['pursuer_positions']):
        p.position = pos
    for e, pos in zip(env.evaders, snap['evader_positions']):
        e.position = pos
    env.timestep = snap['timestep']
    env.done = snap['done']
    env.captured = snap['captured']


# ============================================================
# Trainer
# ============================================================

class SelfPlayTrainer:
    """Two-phase self-play training with CTDE for B-SRM-CHFA."""

    def __init__(
        self,
        grid_size: int = 25,
        n_pursuers: int = 3,
        n_evaders: int = 2,
        max_episodes: int = 1000,
        max_steps: int = 100,
        eval_interval: int = 50,
        save_interval: int = 250,
        phase1_n_episodes: int = 16,
        phase1_n_timesteps: int = 8,
        lambda_opp: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'checkpoints',
    ):
        self.grid_size = grid_size
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.phase1_n_episodes = phase1_n_episodes
        self.phase1_n_timesteps = phase1_n_timesteps
        self.lambda_opp = lambda_opp
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Environment
        self.env = PursuitEvasionEnv(
            grid_size=grid_size, n_pursuers=n_pursuers,
            n_evaders=n_evaders, max_steps=max_steps, seed=42,
        )

        # Agents (one per team, shared across team members)
        self.pursuer_agent = BSRMCHFAAgent(grid_size=grid_size, device=device)
        self.evader_agent = BSRMCHFAAgent(grid_size=grid_size, device=device)

        # Phase 1 buffers (raw observation histories + CTDE ground truth)
        self.pursuer_raw_buffer = RawExperienceBuffer(capacity=200)
        self.evader_raw_buffer = RawExperienceBuffer(capacity=200)

        # Value network buffers
        self.pursuer_value_buffer = ValueBuffer(capacity=50_000)
        self.evader_value_buffer = ValueBuffer(capacity=50_000)

        # Metrics
        self.metrics = {
            'episode': [], 'capture_rate': [], 'avg_capture_time': [],
            'avg_cost': [], 'timeout_rate': [],
            'belief_loss_p': [], 'belief_loss_e': [],
            'regret_loss_p': [], 'regret_loss_e': [],
            'value_loss_p': [], 'value_loss_e': [],
        }

    # ============================================================
    # Episode Rollout (with CTDE data collection)
    # ============================================================

    def run_episode(self, episode_num: int) -> Dict:
        """Run one episode, collecting raw data + env snapshots for both phases."""
        pursuer_obs_raw, evader_obs_raw = self.env.reset()
        self.pursuer_agent.reset_episode()
        self.evader_agent.reset_episode()

        epsilon = max(0.05, 1.0 - episode_num / 500)

        # Per-team raw data (for Phase 1 raw buffer)
        p_data = {'obs_dicts': [], 'actions': [], 'costs': [],
                  'true_opponent_positions': [], 'opponent_actions': []}
        e_data = {'obs_dicts': [], 'actions': [], 'costs': [],
                  'true_opponent_positions': [], 'opponent_actions': []}

        # Per-step data (for Phase 2 CF regret computation)
        env_snapshots = []
        all_p_actions = []
        all_e_actions = []
        all_p_costs = []
        all_e_costs = []

        done = False
        timestep = 0

        while not done and timestep < self.max_steps:
            p_dicts = [obs_to_dict(o, self.grid_size) for o in pursuer_obs_raw]
            e_dicts = [obs_to_dict(o, self.grid_size) for o in evader_obs_raw]

            # Save env state BEFORE actions (for CF rollback)
            env_snapshots.append(save_env_state(self.env))

            # Record true positions (CTDE privileged info)
            true_evader_pos = [e.position for e in self.env.evaders]
            true_pursuer_pos = [p.position for p in self.env.pursuers]

            # Select actions
            p_actions = [self.pursuer_agent.select_action(obs, epsilon=epsilon)
                         for obs in p_dicts]
            e_actions = [self.evader_agent.select_action(obs, epsilon=epsilon)
                         for obs in e_dicts]

            # Record raw data for Phase 1 (first agent per team)
            p_data['obs_dicts'].append(p_dicts[0])
            p_data['actions'].append(p_actions[0])
            p_data['true_opponent_positions'].append(true_evader_pos)
            p_data['opponent_actions'].append(e_actions[0])

            e_data['obs_dicts'].append(e_dicts[0])
            e_data['actions'].append(e_actions[0])
            e_data['true_opponent_positions'].append(true_pursuer_pos)
            e_data['opponent_actions'].append(p_actions[0])

            # Store transitions for agent history (used by select_action)
            self.pursuer_agent.store_transition(p_dicts[0], p_actions[0], 1.0)
            self.evader_agent.store_transition(e_dicts[0], e_actions[0], -1.0)

            all_p_actions.append(p_actions)
            all_e_actions.append(e_actions)

            # Step environment
            pursuer_obs_raw, evader_obs_raw, p_costs, e_costs, done = \
                self.env.step(p_actions, e_actions)

            p_data['costs'].append(p_costs[0])
            e_data['costs'].append(e_costs[0])
            all_p_costs.append(p_costs)
            all_e_costs.append(e_costs)
            timestep += 1

        # Store in raw buffers for future Phase 1 sampling
        self.pursuer_raw_buffer.add(p_data)
        self.evader_raw_buffer.add(e_data)

        return {
            'p_data': p_data, 'e_data': e_data,
            'env_snapshots': env_snapshots,
            'all_p_actions': all_p_actions, 'all_e_actions': all_e_actions,
            'all_p_costs': all_p_costs, 'all_e_costs': all_e_costs,
            'timesteps': timestep, 'captured': self.env.captured, 'done': done,
        }

    # ============================================================
    # Phase 1: Representation Update
    # ============================================================

    def phase1_update(self, agent: BSRMCHFAAgent, raw_buffer: RawExperienceBuffer) -> Dict:
        """Train encoder + belief engine + opponent predictor WITH gradients.

        Samples raw observation histories from the buffer and uses CTDE
        ground truth (true opponent positions) as supervision targets.
        Gradients flow: loss → belief_engine → encoder (end-to-end).
        """
        if len(raw_buffer) < 1:
            return {}

        episodes = raw_buffer.sample(min(self.phase1_n_episodes, len(raw_buffer)))

        total_belief_loss = torch.tensor(0.0, device=self.device)
        total_opp_loss = torch.tensor(0.0, device=self.device)
        n_belief = 0
        n_opp = 0

        for ep_data in episodes:
            T = len(ep_data['obs_dicts'])
            if T < 2:
                continue

            # Sample random timesteps from this episode
            sample_t = sorted(np.random.choice(
                T, min(self.phase1_n_timesteps, T), replace=False
            ))

            for t in sample_t:
                # Build history up to time t
                history_until_t = [
                    {'obs': ep_data['obs_dicts'][i],
                     'action': ep_data['actions'][i],
                     'cost': ep_data['costs'][i]}
                    for i in range(t + 1)
                ]

                # Forward pass WITH gradients through encoder
                z_t, _ = agent.history_encoder(history_until_t)

                # --- Belief loss (supervised against true opponent positions) ---
                # Get raw logits from belief engine (bypass softmax)
                mu, logvar = agent.belief_engine.encode(z_t)
                z_latent = agent.belief_engine.reparameterize(mu, logvar)
                belief_logits = agent.belief_engine.decoder(z_latent)  # (1, grid*grid)

                # Create soft target: equal weight on all opponent positions
                true_positions = ep_data['true_opponent_positions'][t]
                target_dist = torch.zeros(self.grid_size * self.grid_size,
                                          device=self.device)
                for (ox, oy) in true_positions:
                    cell = oy * self.grid_size + ox
                    if 0 <= cell < self.grid_size * self.grid_size:
                        target_dist[cell] = 1.0 / len(true_positions)

                # KL divergence: target || softmax(logits)
                log_probs = F.log_softmax(belief_logits.squeeze(0), dim=0)
                belief_loss = F.kl_div(log_probs, target_dist, reduction='sum')
                total_belief_loss = total_belief_loss + belief_loss
                n_belief += 1

                # --- Opponent prediction loss (predict next position) ---
                if t + 1 < T:
                    opp_logits = agent.opponent_predictor.predictor(z_t)  # (1, grid*grid)
                    next_positions = ep_data['true_opponent_positions'][t + 1]
                    opp_target = torch.zeros(self.grid_size * self.grid_size,
                                             device=self.device)
                    for (ox, oy) in next_positions:
                        cell = oy * self.grid_size + ox
                        if 0 <= cell < self.grid_size * self.grid_size:
                            opp_target[cell] = 1.0 / len(next_positions)

                    opp_log_probs = F.log_softmax(opp_logits.squeeze(0), dim=0)
                    opp_loss = F.kl_div(opp_log_probs, opp_target, reduction='sum')
                    total_opp_loss = total_opp_loss + opp_loss
                    n_opp += 1

        if n_belief == 0:
            return {}

        avg_belief = total_belief_loss / n_belief
        avg_opp = total_opp_loss / max(n_opp, 1)
        combined = avg_belief + self.lambda_opp * avg_opp

        # Backprop through encoder + belief + opponent (end-to-end)
        agent.encoder_optimizer.zero_grad()
        agent.belief_optimizer.zero_grad()
        agent.opponent_optimizer.zero_grad()
        combined.backward()
        torch.nn.utils.clip_grad_norm_(agent.history_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(agent.belief_engine.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(agent.opponent_predictor.parameters(), 1.0)
        agent.encoder_optimizer.step()
        agent.belief_optimizer.step()
        agent.opponent_optimizer.step()

        return {
            'belief_loss': avg_belief.item(),
            'opponent_loss': avg_opp.item(),
        }

    # ============================================================
    # Phase 2: Regret Update
    # ============================================================

    def phase2_compute_regrets(self, agent: BSRMCHFAAgent, ep_data: Dict,
                                env_snapshots: List, all_team_actions: List,
                                all_opponent_actions: List, all_costs: List,
                                is_pursuer: bool) -> int:
        """Compute counterfactual regrets via env rollback + MC cost-to-go.

        For each timestep t:
          baseline = actual MC cost-to-go from t
          For each alternative action a:
            Restore env → simulate 1 step with a → cf_cost + remaining
          regret(a) = baseline - cf_value(a)

        Stores (z_t.detach(), belief.detach(), action, regret) in reservoir.
        """
        T = len(ep_data['obs_dicts'])
        if T < 1:
            return 0

        # Pre-compute MC cost-to-go
        cost_to_go = [0.0] * (T + 1)
        for t in range(T - 1, -1, -1):
            cost_to_go[t] = ep_data['costs'][t] + cost_to_go[t + 1]

        total_added = 0
        n_actions = 5

        for t in range(T):
            # Re-encode history up to t with the UPDATED encoder (but detached)
            history_until_t = [
                {'obs': ep_data['obs_dicts'][i],
                 'action': ep_data['actions'][i],
                 'cost': ep_data['costs'][i]}
                for i in range(t + 1)
            ]

            with torch.no_grad():
                z_t, _ = agent.target_encoder(history_until_t)
                belief = agent.belief_engine.get_belief(z_t)

            baseline = cost_to_go[t]
            remaining = cost_to_go[t + 1] if t + 1 < T else 0.0

            for a in range(n_actions):
                # Restore env to state at time t
                restore_env_state(self.env, env_snapshots[t])

                # Build counterfactual actions
                cf_team = list(all_team_actions[t])
                cf_team[0] = a  # replace first agent's action

                if is_pursuer:
                    _, _, cf_costs, _, cf_done = self.env.step(
                        cf_team, all_opponent_actions[t])
                    cf_cost = cf_costs[0]
                else:
                    _, _, _, cf_costs, cf_done = self.env.step(
                        all_opponent_actions[t], cf_team)
                    cf_cost = cf_costs[0]

                cf_value = cf_cost if cf_done else cf_cost + remaining
                regret = baseline - cf_value

                agent.reservoir.add(
                    z_t=z_t.squeeze(0),
                    belief=belief.squeeze(0),
                    action=a,
                    regret=regret,
                )
                total_added += 1

        # Mark env done so next episode calls reset
        self.env.done = True
        return total_added

    def phase2_update_regret(self, agent: BSRMCHFAAgent, batch_size: int = 256) -> Dict:
        """Train regret network from reservoir (detached z_t, no encoder gradients)."""
        if len(agent.reservoir) < batch_size:
            return {}

        losses = agent.update(batch_size=batch_size)
        return losses if losses else {}

    # ============================================================
    # Value Network Update
    # ============================================================

    def update_value_network(self, agent: BSRMCHFAAgent, ep_data: Dict,
                              value_buffer: ValueBuffer) -> Dict:
        """Train value network on MC returns from the current episode."""
        T = len(ep_data['obs_dicts'])
        if T < 2:
            return {}

        # Compute MC returns
        cost_to_go = [0.0] * (T + 1)
        for t in range(T - 1, -1, -1):
            cost_to_go[t] = ep_data['costs'][t] + cost_to_go[t + 1]

        # Store (z_t, mc_return) in value buffer
        for t in range(T):
            history_until_t = [
                {'obs': ep_data['obs_dicts'][i],
                 'action': ep_data['actions'][i],
                 'cost': ep_data['costs'][i]}
                for i in range(t + 1)
            ]
            with torch.no_grad():
                z_t, _ = agent.history_encoder(history_until_t)
            value_buffer.add(z_t, cost_to_go[t])

        # Train value network
        if len(value_buffer) < 64:
            return {}

        z_batch, return_batch = value_buffer.sample(min(128, len(value_buffer)),
                                                     self.device)
        v_pred = agent.value_network(z_batch)
        v_loss = F.mse_loss(v_pred, return_batch)

        agent.value_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.value_network.parameters(), 1.0)
        agent.value_optimizer.step()

        return {'value_loss': v_loss.item()}

    # ============================================================
    # Evaluation
    # ============================================================

    def evaluate(self, num_episodes: int = 50) -> Dict:
        """Evaluate current policies without exploration."""
        captures, timeouts, capture_times, costs = 0, 0, [], []

        for _ in range(num_episodes):
            p_obs_raw, e_obs_raw = self.env.reset()
            self.pursuer_agent.reset_episode()
            self.evader_agent.reset_episode()

            done = False
            t = 0
            ep_cost = 0.0

            while not done and t < self.max_steps:
                p_dicts = [obs_to_dict(o, self.grid_size) for o in p_obs_raw]
                e_dicts = [obs_to_dict(o, self.grid_size) for o in e_obs_raw]

                p_actions = [self.pursuer_agent.select_action(obs, epsilon=0.0)
                             for obs in p_dicts]
                e_actions = [self.evader_agent.select_action(obs, epsilon=0.0)
                             for obs in e_dicts]

                self.pursuer_agent.store_transition(p_dicts[0], p_actions[0], 1.0)
                self.evader_agent.store_transition(e_dicts[0], e_actions[0], -1.0)

                p_obs_raw, e_obs_raw, p_costs, e_costs, done = \
                    self.env.step(p_actions, e_actions)
                ep_cost += p_costs[0]
                t += 1

            if self.env.captured:
                captures += 1
                capture_times.append(t)
            if t >= self.max_steps and not self.env.captured:
                timeouts += 1
            costs.append(ep_cost)

        return {
            'capture_rate': captures / num_episodes,
            'timeout_rate': timeouts / num_episodes,
            'avg_capture_time': np.mean(capture_times) if capture_times else 0.0,
            'avg_cost': np.mean(costs),
        }

    # ============================================================
    # Main Training Loop
    # ============================================================

    def train(self):
        """Two-phase training loop."""
        print(f"B-SRM-CHFA Two-Phase Self-Play Training")
        print(f"  Episodes: {self.max_episodes} | Grid: {self.grid_size}x{self.grid_size}")
        print(f"  Pursuers: {self.n_pursuers} | Evaders: {self.n_evaders}")
        print(f"  Device: {self.device}")
        print(f"  Phase 1: {self.phase1_n_episodes} eps × {self.phase1_n_timesteps} timesteps")
        print("=" * 60)

        for episode in tqdm(range(self.max_episodes), desc="Training"):
            # ---- ROLLOUT ----
            traj = self.run_episode(episode)

            # ---- PHASE 1: Representation Update (encoder + belief + opponent) ----
            p1_p = self.phase1_update(self.pursuer_agent, self.pursuer_raw_buffer)
            p1_e = self.phase1_update(self.evader_agent, self.evader_raw_buffer)

            # ---- PHASE 2: Regret Update ----
            # 2a: Compute counterfactual regrets via env rollback
            self.phase2_compute_regrets(
                self.pursuer_agent, traj['p_data'], traj['env_snapshots'],
                traj['all_p_actions'], traj['all_e_actions'],
                traj['all_p_costs'], is_pursuer=True,
            )
            self.phase2_compute_regrets(
                self.evader_agent, traj['e_data'], traj['env_snapshots'],
                traj['all_e_actions'], traj['all_p_actions'],
                traj['all_e_costs'], is_pursuer=False,
            )

            # 2b: Train regret network from reservoir
            p2_p = self.phase2_update_regret(self.pursuer_agent)
            p2_e = self.phase2_update_regret(self.evader_agent)

            # ---- VALUE NETWORK UPDATE ----
            v_p = self.update_value_network(
                self.pursuer_agent, traj['p_data'], self.pursuer_value_buffer)
            v_e = self.update_value_network(
                self.evader_agent, traj['e_data'], self.evader_value_buffer)

            # ---- TARGET ENCODER SOFT UPDATE ----
            self.pursuer_agent.soft_update_target_encoder()
            self.evader_agent.soft_update_target_encoder()

            # ---- Evaluation ----
            if (episode + 1) % self.eval_interval == 0:
                eval_m = self.evaluate(num_episodes=50)
                self.metrics['episode'].append(episode + 1)
                self.metrics['capture_rate'].append(eval_m['capture_rate'])
                self.metrics['avg_capture_time'].append(eval_m['avg_capture_time'])
                self.metrics['avg_cost'].append(eval_m['avg_cost'])
                self.metrics['timeout_rate'].append(eval_m['timeout_rate'])
                self.metrics['belief_loss_p'].append(p1_p.get('belief_loss', 0))
                self.metrics['belief_loss_e'].append(p1_e.get('belief_loss', 0))
                self.metrics['regret_loss_p'].append(p2_p.get('regret_loss', 0))
                self.metrics['regret_loss_e'].append(p2_e.get('regret_loss', 0))
                self.metrics['value_loss_p'].append(v_p.get('value_loss', 0))
                self.metrics['value_loss_e'].append(v_e.get('value_loss', 0))

                tqdm.write(
                    f"\n  [Ep {episode+1}/{self.max_episodes} | "
                    f"Cap={eval_m['capture_rate']:.0%} | "
                    f"AvgTime={eval_m['avg_capture_time']:.1f} | "
                    f"Cost={eval_m['avg_cost']:.1f} | "
                    f"TO={eval_m['timeout_rate']:.0%} | "
                    f"BelLoss={p1_p.get('belief_loss', 0):.3f} | "
                    f"RegLoss={p2_p.get('regret_loss', 0):.4f}]"
                )

            # ---- Checkpoint ----
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(episode + 1)

        # Final save
        self.save_checkpoint(self.max_episodes)
        self.plot_metrics()
        print("\nTraining completed!")

    # ============================================================
    # Checkpointing & Plotting
    # ============================================================

    def save_checkpoint(self, episode: int):
        checkpoint = {
            'episode': episode,
            'pursuer_encoder': self.pursuer_agent.history_encoder.state_dict(),
            'pursuer_belief': self.pursuer_agent.belief_engine.state_dict(),
            'pursuer_regret': self.pursuer_agent.regret_minimizer.state_dict(),
            'pursuer_opponent': self.pursuer_agent.opponent_predictor.state_dict(),
            'pursuer_value': self.pursuer_agent.value_network.state_dict(),
            'evader_encoder': self.evader_agent.history_encoder.state_dict(),
            'evader_belief': self.evader_agent.belief_engine.state_dict(),
            'evader_regret': self.evader_agent.regret_minimizer.state_dict(),
            'evader_opponent': self.evader_agent.opponent_predictor.state_dict(),
            'evader_value': self.evader_agent.value_network.state_dict(),
            'metrics': self.metrics,
        }
        save_path = self.save_dir / f'checkpoint_v2_ep{episode}.pt'
        torch.save(checkpoint, save_path)
        tqdm.write(f"  Saved checkpoint → {save_path}")

    def plot_metrics(self):
        if not self.metrics['episode']:
            print("No metrics to plot.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(14, 14))

        # Row 1: Game performance
        axes[0, 0].plot(self.metrics['episode'], self.metrics['capture_rate'])
        axes[0, 0].set_ylabel('Capture Rate')
        axes[0, 0].set_title('Capture Rate')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.metrics['episode'], self.metrics['avg_capture_time'])
        axes[0, 1].set_ylabel('Avg Capture Time')
        axes[0, 1].set_title('Average Capture Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Row 2: Phase 1 losses
        axes[1, 0].plot(self.metrics['episode'], self.metrics['belief_loss_p'], label='Pursuer')
        axes[1, 0].plot(self.metrics['episode'], self.metrics['belief_loss_e'], label='Evader')
        axes[1, 0].set_ylabel('Belief Loss')
        axes[1, 0].set_title('Phase 1: Belief Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.metrics['episode'], self.metrics['value_loss_p'], label='Pursuer')
        axes[1, 1].plot(self.metrics['episode'], self.metrics['value_loss_e'], label='Evader')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Network Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Row 3: Phase 2 losses + cost
        axes[2, 0].plot(self.metrics['episode'], self.metrics['regret_loss_p'], label='Pursuer')
        axes[2, 0].plot(self.metrics['episode'], self.metrics['regret_loss_e'], label='Evader')
        axes[2, 0].set_ylabel('Regret Loss')
        axes[2, 0].set_title('Phase 2: Regret Loss')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(self.metrics['episode'], self.metrics['avg_cost'])
        axes[2, 1].set_ylabel('Avg Cost')
        axes[2, 1].set_title('Average Cost')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves_v2.png', dpi=150)
        print(f"Saved training curves → {self.save_dir / 'training_curves_v2.png'}")

        with open(self.save_dir / 'metrics_v2.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)


# ============================================================
# Entry Point
# ============================================================

def main():
    trainer = SelfPlayTrainer(
        grid_size=25,
        n_pursuers=3,
        n_evaders=2,
        max_episodes=1000,
        max_steps=100,
        eval_interval=50,
        save_interval=250,
        phase1_n_episodes=16,
        phase1_n_timesteps=8,
        lambda_opp=0.1,
        save_dir='checkpoints',
    )
    trainer.train()


if __name__ == '__main__':
    main()
