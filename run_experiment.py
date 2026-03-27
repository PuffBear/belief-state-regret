"""
B-SRM-CHFA Experiment Runner

CLI entry point for training and evaluating agents on pursuit-evasion.

Usage:
    python run_experiment.py --agent bsrmchfa --grid-size 10 --episodes 10000
    python run_experiment.py --agent greedy   --grid-size 10 --episodes 100
    python run_experiment.py --agent random   --grid-size 10 --episodes 100
    python run_experiment.py --agent qlearning --grid-size 10 --episodes 100 --train-episodes 5000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import time


def run_bsrmchfa(args):
    """Train and evaluate B-SRM-CHFA agent."""
    from training.self_play import Trainer

    print("=" * 60)
    print("B-SRM-CHFA Training")
    print("=" * 60)

    device = "cuda" if args.device == "auto" else args.device
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, falling back to CPU")
    except ImportError:
        device = "cpu"

    trainer = Trainer(
        grid_size=args.grid_size,
        n_pursuers=args.n_pursuers,
        n_evaders=args.n_evaders,
        max_steps=args.max_steps,
        embedding_dim=args.embedding_dim,
        reservoir_capacity=args.reservoir_capacity,
        lr=args.lr,
        batch_size=args.batch_size,
        update_frequency=args.update_frequency,
        eval_interval=args.eval_interval,
        device=device,
        evader_policy=args.evader_policy,
        seed=args.seed,
    )

    start_time = time.time()
    metrics = trainer.train(num_episodes=args.episodes)
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (100 episodes)")
    print("=" * 60)

    eval_results = trainer.evaluate(n_trials=100)
    print(f"  Avg capture time: {eval_results['avg_timesteps']:.1f} ± {eval_results['std_timesteps']:.1f}")
    print(f"  Capture rate:     {eval_results['capture_rate']:.1%}")
    print(f"  Avg total cost:   {eval_results['avg_cost']:.1f}")

    # Save checkpoint
    if args.save_checkpoint:
        ckpt_path = os.path.join(
            os.path.dirname(__file__),
            f"checkpoint_{args.grid_size}x{args.grid_size}_ep{args.episodes}.pt"
        )
        trainer.save_checkpoint(ckpt_path)

    return eval_results


def run_baseline(args):
    """Evaluate a baseline agent."""
    from baselines.agents import RandomAgent, GreedyPursuer, QLearningAgent, evaluate_baseline

    agent_map = {
        "random": RandomAgent,
        "greedy": GreedyPursuer,
        "qlearning": QLearningAgent,
    }

    cls = agent_map[args.agent]

    print("=" * 60)
    print(f"{args.agent.title()} Baseline Evaluation")
    print("=" * 60)

    train_eps = args.train_episodes if args.agent == "qlearning" else 0

    results = evaluate_baseline(
        cls,
        grid_size=args.grid_size,
        n_pursuers=args.n_pursuers,
        n_evaders=args.n_evaders,
        max_steps=args.max_steps,
        n_episodes=args.episodes,
        train_episodes=train_eps,
        seed=args.seed,
    )

    print(f"  Avg capture time: {results['avg_timesteps']:.1f} ± {results['std_timesteps']:.1f}")
    print(f"  Capture rate:     {results['capture_rate']:.1%}")
    print(f"  Avg total cost:   {results['avg_cost']:.1f}")

    return results


def run_comparison(args):
    """Run all agents and compare results."""
    from baselines.agents import RandomAgent, GreedyPursuer, QLearningAgent, evaluate_baseline
    from training.self_play import Trainer

    print("=" * 60)
    print(f"Agent Comparison on {args.grid_size}×{args.grid_size} Grid")
    print("=" * 60)

    results_all = {}

    # Baselines
    for name, cls, train_eps in [
        ("Random", RandomAgent, 0),
        ("Greedy", GreedyPursuer, 0),
        ("Q-Learning", QLearningAgent, 2000),
    ]:
        print(f"\n--- {name} ---")
        r = evaluate_baseline(
            cls,
            grid_size=args.grid_size,
            n_pursuers=args.n_pursuers,
            n_evaders=args.n_evaders,
            max_steps=args.max_steps,
            n_episodes=100,
            train_episodes=train_eps,
            seed=args.seed,
        )
        results_all[name] = r
        print(f"  Time={r['avg_timesteps']:.1f}±{r['std_timesteps']:.1f}  "
              f"Cap={r['capture_rate']:.1%}  Cost={r['avg_cost']:.1f}")

    # B-SRM-CHFA
    print(f"\n--- B-SRM-CHFA (training {args.episodes} episodes) ---")
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    trainer = Trainer(
        grid_size=args.grid_size,
        n_pursuers=args.n_pursuers,
        n_evaders=args.n_evaders,
        max_steps=args.max_steps,
        device=device,
        seed=args.seed,
    )
    trainer.train(num_episodes=args.episodes)
    r = trainer.evaluate(n_trials=100)
    results_all["B-SRM-CHFA"] = r
    print(f"  Time={r['avg_timesteps']:.1f}±{r['std_timesteps']:.1f}  "
          f"Cap={r['capture_rate']:.1%}  Cost={r['avg_cost']:.1f}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Agent':<15} {'Avg Time':>10} {'Cap Rate':>10} {'Avg Cost':>10}")
    print("-" * 60)
    for name, r in results_all.items():
        print(f"{name:<15} {r['avg_timesteps']:>7.1f}±{r['std_timesteps']:<4.1f} "
              f"{r['capture_rate']:>9.1%} {r['avg_cost']:>10.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="B-SRM-CHFA Experiment Runner")

    parser.add_argument("--agent", type=str, default="bsrmchfa",
                        choices=["bsrmchfa", "random", "greedy", "qlearning", "compare"],
                        help="Agent to train/evaluate")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid dimension")
    parser.add_argument("--n-pursuers", type=int, default=3, help="Number of pursuers")
    parser.add_argument("--n-evaders", type=int, default=2, help="Number of evaders")
    parser.add_argument("--max-steps", type=int, default=100, help="Episode horizon H_max")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # B-SRM-CHFA specific
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--reservoir-capacity", type=int, default=500_000, help="Reservoir buffer capacity")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--update-frequency", type=int, default=10, help="Network update interval")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation logging interval")
    parser.add_argument("--evader-policy", type=str, default="random",
                        choices=["random", "flee"], help="Evader behavior")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save model checkpoint")

    # Q-learning specific
    parser.add_argument("--train-episodes", type=int, default=5000,
                        help="Training episodes for Q-learning before evaluation")

    args = parser.parse_args()

    if args.agent == "compare":
        run_comparison(args)
    elif args.agent == "bsrmchfa":
        run_bsrmchfa(args)
    else:
        run_baseline(args)


if __name__ == "__main__":
    main()
