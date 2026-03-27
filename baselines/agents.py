"""
Baseline Agents for B-SRM-CHFA Comparison

Three baselines:
  1. RandomAgent   — uniform random actions
  2. GreedyPursuer — move toward nearest visible opponent
  3. QLearningAgent — tabular Q-learning with epsilon-greedy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple
from environments.pursuit_evasion_env import Observation, Action


class RandomAgent:
    """Selects actions uniformly at random."""

    def __init__(self, n_actions: int = 5):
        self.n_actions = n_actions

    def select_action(self, obs: Observation) -> Action:
        return Action(np.random.randint(0, self.n_actions))

    def update(self, *args, **kwargs):
        pass  # no learning


class GreedyPursuer:
    """
    Moves toward the nearest detected opponent.
    Falls back to random movement if no opponents visible.
    """

    def __init__(self, n_actions: int = 5):
        self.n_actions = n_actions

    def select_action(self, obs: Observation) -> Action:
        if not obs.detected_opponents:
            return Action(np.random.randint(0, self.n_actions))

        # Find nearest opponent (minimum Manhattan distance)
        best_dx, best_dy = 0, 0
        best_dist = float("inf")
        for dx, dy in obs.detected_opponents:
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                best_dx, best_dy = dx, dy

        # Move toward opponent
        if abs(best_dx) >= abs(best_dy):
            return Action.EAST if best_dx > 0 else Action.WEST
        else:
            return Action.SOUTH if best_dy > 0 else Action.NORTH

    def update(self, *args, **kwargs):
        pass


class QLearningAgent:
    """
    Tabular Q-learning with epsilon-greedy exploration.

    State representation: (self_position, nearest_opponent_direction)
    Simplified for tractability on small grids.
    """

    def __init__(
        self,
        n_actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self._prev_state = None
        self._prev_action = None

    def _obs_to_state(self, obs: Observation) -> tuple:
        """Convert observation to a hashable state key."""
        # Use flattened local grid as state (simplified)
        grid_key = tuple(obs.local_grid.flatten().tolist())

        # Also encode opponent direction
        if obs.detected_opponents:
            nearest = min(obs.detected_opponents, key=lambda d: abs(d[0]) + abs(d[1]))
            opp_dir = (np.sign(nearest[0]), np.sign(nearest[1]))
        else:
            opp_dir = (0, 0)

        return (opp_dir, obs.timestep % 10)  # compact state

    def select_action(self, obs: Observation) -> Action:
        state = self._obs_to_state(obs)

        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = int(np.argmax(self.q_table[state]))

        self._prev_state = state
        self._prev_action = action
        return Action(action)

    def update(self, obs: Observation, reward: float, done: bool):
        """Q-learning update. Note: reward = -cost (we negate costs for Q-learning)."""
        if self._prev_state is None:
            return

        state = self._obs_to_state(obs)
        best_next = np.max(self.q_table[state]) if not done else 0.0

        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[self._prev_state][self._prev_action]
        self.q_table[self._prev_state][self._prev_action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -----------------------------------------------------------------------
# Evaluation harness for baselines
# -----------------------------------------------------------------------

def evaluate_baseline(
    agent_class,
    grid_size: int = 10,
    n_pursuers: int = 3,
    n_evaders: int = 2,
    max_steps: int = 100,
    n_episodes: int = 100,
    train_episodes: int = 0,
    seed: Optional[int] = None,
) -> dict:
    """
    Evaluate a baseline agent over multiple episodes.

    Returns dict with avg_timesteps, capture_rate, avg_cost.
    """
    from environments.pursuit_evasion_env import PursuitEvasionEnv

    env = PursuitEvasionEnv(
        grid_size=grid_size,
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        max_steps=max_steps,
        seed=seed,
    )

    # Create agents
    agents = [agent_class() for _ in range(n_pursuers)]
    evader_agents = [RandomAgent() for _ in range(n_evaders)]

    # Optional training phase (for Q-learning)
    if train_episodes > 0:
        for ep in range(train_episodes):
            p_obs, e_obs = env.reset()
            for t in range(max_steps):
                p_actions = [a.select_action(o) for a, o in zip(agents, p_obs)]
                e_actions = [a.select_action(o) for a, o in zip(evader_agents, e_obs)]
                p_obs, e_obs, p_costs, e_costs, done = env.step(p_actions, e_actions)
                for a, o, c in zip(agents, p_obs, p_costs):
                    a.update(o, -c, done)
                if done:
                    break

    # Evaluation
    results = {"timesteps": [], "captures": [], "costs": []}

    for ep in range(n_episodes):
        p_obs, e_obs = env.reset()
        total_cost = 0.0

        for t in range(max_steps):
            p_actions = [a.select_action(o) for a, o in zip(agents, p_obs)]
            e_actions = [a.select_action(o) for a, o in zip(evader_agents, e_obs)]
            p_obs, e_obs, p_costs, e_costs, done = env.step(p_actions, e_actions)
            total_cost += sum(p_costs)
            if done:
                break

        results["timesteps"].append(env.timestep)
        results["captures"].append(env.captured)
        results["costs"].append(total_cost)

    return {
        "avg_timesteps": np.mean(results["timesteps"]),
        "std_timesteps": np.std(results["timesteps"]),
        "capture_rate": np.mean(results["captures"]),
        "avg_cost": np.mean(results["costs"]),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Agent Evaluation (10×10 grid, 100 episodes each)")
    print("=" * 60)

    for name, cls, train_eps in [
        ("Random", RandomAgent, 0),
        ("Greedy", GreedyPursuer, 0),
        ("Q-Learning", QLearningAgent, 1000),
    ]:
        print(f"\n--- {name} Agent ---")
        results = evaluate_baseline(
            cls,
            grid_size=10,
            n_episodes=100,
            train_episodes=train_eps,
            seed=42,
        )
        print(f"  Avg capture time: {results['avg_timesteps']:.1f} ± {results['std_timesteps']:.1f}")
        print(f"  Capture rate:     {results['capture_rate']:.1%}")
        print(f"  Avg total cost:   {results['avg_cost']:.1f}")

    print("\n✓ Baseline evaluation complete!")
