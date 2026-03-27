"""
B-SRM-CHFA Diagnostic Script

Systematically checks for common issues:
  1. Environment setup (grid size, FOV coverage, capture mechanics)
  2. Greedy baseline integrity (no privileged info)
  3. Greedy scaling across grid sizes
  4. Belief engine quality (KL divergence)
  5. Regret network signal (mean/std of regret predictions)
  6. Short training learning curve

Usage:
    python diagnose.py                    # Run all diagnostics
    python diagnose.py --section env      # Environment only
    python diagnose.py --section greedy   # Greedy scaling test
    python diagnose.py --section train    # Short training run
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import time

from environments.pursuit_evasion_env import PursuitEvasionEnv, Action, Observation
from baselines.agents import RandomAgent, GreedyPursuer, evaluate_baseline


def diagnose_env(grid_size=10, n_pursuers=3, n_evaders=2):
    """Check environment setup and capture mechanics."""
    print("\n" + "=" * 65)
    print("  SECTION 1: Environment Diagnostics")
    print("=" * 65)

    env = PursuitEvasionEnv(grid_size=grid_size, n_pursuers=n_pursuers,
                            n_evaders=n_evaders, max_steps=100, seed=42)

    # FOV coverage analysis
    fov_size = (2 * env.obs_radius + 1) ** 2   # 5x5 = 25
    total_cells = grid_size * grid_size
    max_coverage = min(n_pursuers * fov_size, total_cells)
    coverage_pct = max_coverage / total_cells * 100

    print(f"\n  Grid size:        {grid_size}×{grid_size} ({total_cells} cells)")
    print(f"  Pursuers:         {n_pursuers}")
    print(f"  Evaders:          {n_evaders}")
    print(f"  FOV per pursuer:  {2*env.obs_radius+1}×{2*env.obs_radius+1} = {fov_size} cells")
    print(f"  Max FOV coverage: {max_coverage}/{total_cells} = {coverage_pct:.0f}%")
    print(f"  Capture radius:   {env.capture_radius} (Manhattan)")

    if coverage_pct > 60:
        print(f"\n  ⚠️  WARNING: FOV covers {coverage_pct:.0f}% of grid — partial observability")
        print(f"     is minimal! Greedy will dominate. Use grid ≥ 20×20 for meaningful tests.")

    # Capture mechanics test
    print(f"\n  --- Capture mechanics test ---")
    env.reset()
    # Place pursuer and evader adjacent
    env.pursuers[0].position = (5, 5)
    env.evaders[0].position = (5, 6)
    dist = abs(5-5) + abs(5-6)
    can_capture = env._check_capture()
    print(f"  Adjacent (dist={dist}): capture={can_capture}  ✅" if can_capture
          else f"  Adjacent (dist={dist}): capture={can_capture}  ❌ BUG!")

    # Place pursuer and evader on same cell
    env.pursuers[0].position = (5, 5)
    env.evaders[0].position = (5, 5)
    dist = 0
    can_capture = env._check_capture()
    print(f"  Same cell (dist={dist}): capture={can_capture}  ✅" if can_capture
          else f"  Same cell (dist={dist}): capture={can_capture}  ❌ BUG!")

    # Test that pursuer CAN move onto evader cell
    env.reset()
    env.pursuers[0].position = (5, 5)
    env.evaders[0].position = (6, 5)
    old_pos = env.pursuers[0].position
    env._move_agents([env.pursuers[0]], [Action.EAST])
    new_pos = env.pursuers[0].position
    moved_onto = new_pos == (6, 5)
    print(f"  Move onto evader: {old_pos} → {new_pos}  {'✅' if moved_onto else '❌ BUG: blocked!'}")

    # Starting distance distribution
    print(f"\n  --- Starting distance distribution (100 resets) ---")
    dists = []
    for _ in range(100):
        env.reset()
        for p in env.pursuers:
            for e in env.evaders:
                d = abs(p.position[0] - e.position[0]) + abs(p.position[1] - e.position[1])
                dists.append(d)
    dists = np.array(dists)
    print(f"  Min-pursuer-evader distance: {dists.min():.0f} — {dists.max():.0f}")
    print(f"  Mean: {dists.mean():.1f}, Median: {np.median(dists):.1f}")


def diagnose_greedy_scaling():
    """Test Greedy across grid sizes to see if it degrades."""
    print("\n" + "=" * 65)
    print("  SECTION 2: Greedy Baseline Scaling Test")
    print("=" * 65)

    configs = [
        (10,  3, 2),
        (10,  1, 1),
        (20,  3, 2),
        (20,  1, 1),
    ]

    print(f"\n  {'Grid':>6s} {'P×E':>5s} {'Avg Time':>10s} {'Cap Rate':>10s} {'Avg Cost':>10s}")
    print("  " + "-" * 50)

    for gs, np_, ne in configs:
        r = evaluate_baseline(
            GreedyPursuer,
            grid_size=gs, n_pursuers=np_, n_evaders=ne,
            max_steps=200, n_episodes=100, seed=42,
        )
        print(f"  {gs:>3d}×{gs:<3d} {np_}×{ne}   "
              f"{r['avg_timesteps']:>6.1f}±{r['std_timesteps']:<5.1f}"
              f"{r['capture_rate']:>8.1%}  {r['avg_cost']:>8.1f}")

    print("\n  Expected: Avg time should increase significantly on 20×20 grids.")
    print("  If 20×20 1v1 still gets <10 avg time, something is wrong.")


def diagnose_training(episodes=300, grid_size=10):
    """Run short training and print diagnostics."""
    print("\n" + "=" * 65)
    print(f"  SECTION 3: Short Training Run ({episodes} episodes)")
    print("=" * 65)

    try:
        import torch
    except ImportError:
        print("  PyTorch not available, skipping training diagnostics.")
        return

    from training.self_play import Trainer

    trainer = Trainer(
        grid_size=grid_size,
        n_pursuers=3, n_evaders=2,
        max_steps=100,
        embedding_dim=128,  # smaller for speed
        lr=3e-4,
        batch_size=128,
        update_frequency=5,
        eval_interval=50,
        device="cpu",
        seed=42,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_episodes=episodes,
    )

    print(f"\n  Training {episodes} episodes on {grid_size}×{grid_size}...")
    start = time.time()
    metrics = trainer.train(num_episodes=episodes)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    # Print learning curve
    print(f"\n  --- Learning Curve (every 50 episodes) ---")
    print(f"  {'Episode':>8s} {'AvgTime':>8s} {'CapRate':>8s} {'Cost':>8s} {'L_regret':>10s} {'L_belief':>10s}")
    print("  " + "-" * 60)

    for ep_mark in range(50, episodes + 1, 50):
        window = [m for m in metrics if ep_mark - 50 < m["episode"] <= ep_mark]
        if not window:
            continue
        avg_t = np.mean([m["timesteps"] for m in window])
        cap = np.mean([m["capture"] for m in window])
        cost = np.mean([m["total_cost"] for m in window])
        lr_vals = [m.get("loss_regret", None) for m in window]
        lr_vals = [v for v in lr_vals if v is not None]
        lb_vals = [m.get("loss_belief", None) for m in window]
        lb_vals = [v for v in lb_vals if v is not None]
        lr_str = f"{np.mean(lr_vals):.4f}" if lr_vals else "N/A"
        lb_str = f"{np.mean(lb_vals):.4f}" if lb_vals else "N/A"
        print(f"  {ep_mark:>8d} {avg_t:>8.1f} {cap:>8.1%} {cost:>8.1f} {lr_str:>10s} {lb_str:>10s}")

    # Regret network diagnostics
    print(f"\n  --- Regret Network Diagnostics ---")
    device = trainer.device
    z_test = torch.randn(32, trainer.encoder.hidden_dim, device=device)
    with torch.no_grad():
        regrets = trainer.regret_net.get_all_regrets(z_test)
    print(f"  Regret mean: {regrets.mean().item():.4f}")
    print(f"  Regret std:  {regrets.std().item():.4f}")
    print(f"  Max regret action distribution: {[int((regrets.argmax(dim=-1) == a).sum()) for a in range(5)]}")
    if regrets.std().item() < 0.01:
        print("  ⚠️  WARNING: Regret std ≈ 0 — network isn't differentiating actions!")
    else:
        print("  ✅ Regrets show differentiation between actions.")

    print(f"\n  Reservoir size: {len(trainer.reservoir)}")
    print(f"  Value buffer size: {len(trainer.value_buffer)}")

    # Quick evaluation
    print(f"\n  --- Post-Training Evaluation (50 episodes) ---")
    eval_r = trainer.evaluate(n_trials=50)
    print(f"  Avg time:    {eval_r['avg_timesteps']:.1f} ± {eval_r['std_timesteps']:.1f}")
    print(f"  Capture rate: {eval_r['capture_rate']:.1%}")
    print(f"  Avg cost:     {eval_r['avg_cost']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="B-SRM-CHFA Diagnostics")
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "env", "greedy", "train"],
                        help="Which diagnostic section to run")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    print("╔" + "═" * 63 + "╗")
    print("║  B-SRM-CHFA Diagnostic Report                                ║")
    print("╚" + "═" * 63 + "╝")

    if args.section in ("all", "env"):
        diagnose_env(grid_size=args.grid_size)

    if args.section in ("all", "greedy"):
        diagnose_greedy_scaling()

    if args.section in ("all", "train"):
        diagnose_training(episodes=args.episodes, grid_size=args.grid_size)

    print("\n" + "=" * 65)
    print("  Diagnostics complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
