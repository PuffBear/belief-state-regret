"""
Quick test script to verify environment and B-SRM-CHFA agent work correctly.
Run this before starting full training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from environments.pursuit_evasion_env import PursuitEvasionEnv, Action
from agents.bsrmchfa_agent import BSRMCHFAAgent


def obs_to_dict(observation, grid_size=25):
    """Convert environment Observation dataclass to the dict format the agent expects."""
    # Extract the local_grid as fov_grid
    fov_grid = observation.local_grid.astype(np.float32)
    
    # own_pos: the agent's actual position isn't directly in the Observation
    # self_position is relative (center of view), so we use it as-is
    own_pos = np.array(observation.self_position, dtype=np.float32)
    
    # last_known_opponent: use detected_opponents if any, else default
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


def test_environment():
    """Test environment basic functionality"""
    print("="*60)
    print("Testing Environment...")
    print("="*60)
    
    env = PursuitEvasionEnv(grid_size=25, n_pursuers=3, n_evaders=2, seed=42)
    
    # Reset
    pursuer_obs, evader_obs = env.reset()
    print(f"\n✓ Environment reset successful")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Pursuers: {env.n_pursuers}, Evaders: {env.n_evaders}")
    print(f"  Walls: {int(env.walls.sum())}")
    
    # Check observations
    print(f"\n✓ Pursuer observation structure:")
    print(f"  Local grid shape: {pursuer_obs[0].local_grid.shape}")
    print(f"  Self position: {pursuer_obs[0].self_position}")
    print(f"  Detected opponents: {pursuer_obs[0].detected_opponents}")
    
    # Run a few steps
    print(f"\n✓ Running 10 random steps...")
    done = False
    step_count = 0
    
    while not done and step_count < 10:
        pursuer_actions = [Action(env.rng.integers(0, 5)) for _ in range(env.n_pursuers)]
        evader_actions = [Action(env.rng.integers(0, 5)) for _ in range(env.n_evaders)]
        
        pursuer_obs, evader_obs, p_costs, e_costs, done = env.step(
            pursuer_actions, evader_actions
        )
        
        step_count += 1
    
    print(f"  Completed {step_count} steps")
    print(f"  Episode ended: {done}")
    print(f"  Captured: {env.captured}")
    
    print("\n✓ Environment test PASSED!\n")
    return True


def test_agent():
    """Test B-SRM-CHFA agent"""
    print("="*60)
    print("Testing B-SRM-CHFA Agent...")
    print("="*60)
    
    agent = BSRMCHFAAgent(grid_size=25, fov_size=5, device='cpu')
    
    print(f"\n✓ Agent created successfully")
    print(f"  Device: cpu")
    print(f"  Grid size: 25x25")
    
    # Create dummy observation (agent expects dict format)
    obs = {
        'fov_grid': np.random.rand(5, 5, 3).astype(np.float32),
        'own_pos': np.array([10, 10], dtype=np.float32),
        'last_known_opponent': np.array([15, 15, 5], dtype=np.float32)
    }
    
    # Reset episode
    agent.reset_episode()
    print(f"\n✓ Episode reset")
    
    # Select action (no history)
    action = agent.select_action(obs)
    print(f"✓ Selected action (empty history): {action}")
    
    # Store transition
    agent.store_transition(obs, action, cost=1.0)
    print(f"✓ Transition stored")
    
    # Select action (with history)
    action2 = agent.select_action(obs)
    print(f"✓ Selected action (with history): {action2}")
    
    # Store more transitions
    for i in range(5):
        obs_new = {
            'fov_grid': np.random.rand(5, 5, 3).astype(np.float32),
            'own_pos': np.array([10 + i, 10 + i], dtype=np.float32),
            'last_known_opponent': np.array([15, 15, 5 + i], dtype=np.float32)
        }
        action = agent.select_action(obs_new)
        agent.store_transition(obs_new, action, cost=1.0)
    
    print(f"✓ Stored 7 transitions total")
    print(f"  History length: {len(agent.history)}")
    
    # Test reservoir
    print(f"\n✓ Reservoir buffer:")
    print(f"  Current size: {len(agent.reservoir)}")
    print(f"  Capacity: {agent.reservoir.capacity}")
    
    print("\n✓ Agent test PASSED!\n")
    return True


def test_integration():
    """Test environment + agent integration"""
    print("="*60)
    print("Testing Environment + Agent Integration...")
    print("="*60)
    
    env = PursuitEvasionEnv(grid_size=25, n_pursuers=3, n_evaders=2, seed=42)
    pursuer_agent = BSRMCHFAAgent(grid_size=25, device='cpu')
    evader_agent = BSRMCHFAAgent(grid_size=25, device='cpu')
    
    # Reset
    pursuer_obs, evader_obs = env.reset()
    pursuer_agent.reset_episode()
    evader_agent.reset_episode()
    
    print(f"\n✓ Environment and agents reset")
    
    # Run episode
    done = False
    timesteps = 0
    
    print(f"\n✓ Running full episode...")
    
    while not done and timesteps < 50:  # Limit to 50 steps for test
        # Convert Observation dataclass to agent-expected dict format
        pursuer_obs_dicts = [obs_to_dict(o) for o in pursuer_obs]
        evader_obs_dicts = [obs_to_dict(o) for o in evader_obs]
        
        # Select actions
        pursuer_actions = [
            pursuer_agent.select_action(obs)
            for obs in pursuer_obs_dicts
        ]
        
        evader_actions = [
            evader_agent.select_action(obs)
            for obs in evader_obs_dicts
        ]
        
        # Store transitions
        pursuer_agent.store_transition(pursuer_obs_dicts[0], pursuer_actions[0], 1.0)
        evader_agent.store_transition(evader_obs_dicts[0], evader_actions[0], -1.0)
        
        # Step
        pursuer_obs, evader_obs, p_costs, e_costs, done = env.step(
            pursuer_actions, evader_actions
        )
        
        timesteps += 1
    
    print(f"  Episode completed in {timesteps} steps")
    print(f"  Captured: {env.captured}")
    print(f"  Timeout: {timesteps >= env.max_steps}")
    print(f"  Pursuer history length: {len(pursuer_agent.history)}")
    print(f"  Evader history length: {len(evader_agent.history)}")
    
    print("\n✓ Integration test PASSED!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("B-SRM-CHFA Quick Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("Environment", test_environment),
        ("Agent", test_agent),
        ("Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED!")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("✓ All tests passed! Ready to start training.\n")
        print("To start training, run:")
        print("  python training/self_play_train.py")
    else:
        print("✗ Some tests failed. Please fix errors before training.\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
