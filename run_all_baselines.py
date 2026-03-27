"""
Run baseline comparisons across all 3 game configurations:
  Config A: 1v1 (single-agent)
  Config B: 3v2 (default multi-agent)
  Config C: 3v3 (larger multi-agent)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from baselines.agents import RandomAgent, GreedyPursuer, QLearningAgent, evaluate_baseline


CONFIGS = [
    {"name": "1v1 (Single-Agent)", "n_pursuers": 1, "n_evaders": 1},
    {"name": "3v2 (Default)",      "n_pursuers": 3, "n_evaders": 2},
    {"name": "3v3 (Large)",        "n_pursuers": 3, "n_evaders": 3},
]

AGENTS = [
    ("Random",     RandomAgent,    0),
    ("Greedy",     GreedyPursuer,  0),
    ("Q-Learning", QLearningAgent, 2000),
]

GRID_SIZE = 10
N_EVAL = 200
SEED = 42


def main():
    all_results = {}

    for cfg in CONFIGS:
        print("\n" + "=" * 65)
        print(f"  Config: {cfg['name']}  ({cfg['n_pursuers']}P vs {cfg['n_evaders']}E, {GRID_SIZE}x{GRID_SIZE})")
        print("=" * 65)

        cfg_results = {}
        for agent_name, agent_cls, train_eps in AGENTS:
            r = evaluate_baseline(
                agent_cls,
                grid_size=GRID_SIZE,
                n_pursuers=cfg["n_pursuers"],
                n_evaders=cfg["n_evaders"],
                max_steps=100,
                n_episodes=N_EVAL,
                train_episodes=train_eps,
                seed=SEED,
            )
            cfg_results[agent_name] = r
            print(f"  {agent_name:<12s}  Time={r['avg_timesteps']:5.1f}±{r['std_timesteps']:<5.1f}"
                  f"  Cap={r['capture_rate']:5.1%}  Cost={r['avg_cost']:6.1f}")

        all_results[cfg["name"]] = cfg_results

    # ---- Summary Table ----
    print("\n\n" + "=" * 75)
    print("  SUMMARY: Baseline Comparison Across All Configurations")
    print("=" * 75)
    print(f"{'Config':<22s} {'Agent':<12s} {'Avg Time':>10s} {'Cap Rate':>10s} {'Avg Cost':>10s}")
    print("-" * 75)

    for cfg_name, agents in all_results.items():
        for agent_name, r in agents.items():
            print(f"{cfg_name:<22s} {agent_name:<12s} "
                  f"{r['avg_timesteps']:>7.1f}±{r['std_timesteps']:<4.1f}"
                  f"{r['capture_rate']:>9.1%} {r['avg_cost']:>10.1f}")
        print("-" * 75)

    print("=" * 75)


if __name__ == "__main__":
    main()
