import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob

from environments.pursuit_evasion_env import PursuitEvasionEnv
from agents.bsrmchfa_agent import BSRMCHFAAgent
from training.self_play_train import obs_to_dict

def compute_nearest_distance(env, evader_idx):
    """Calculates true Manhattan distance to the nearest pursuer for a specific evader."""
    e_pos = env.evaders[evader_idx].position
    
    min_dist = float('inf')
    for p_idx in range(env.n_pursuers):
        p_pos = env.pursuers[p_idx].position
        dist = abs(e_pos[0] - p_pos[0]) + abs(e_pos[1] - p_pos[1])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def run_evaluation_mode(env, pursuer_agent, evader_agent, mode='trained', eval_episodes=500):
    capture_history = []
    episode_lengths = []
    delta_distances = [] # E[Δd_t] inside FOV
    
    desc = f"Evaluating Mode: {'Trained Evaders' if mode == 'trained' else 'Random Evaders'}"
    
    with torch.no_grad():
        for episode in tqdm(range(eval_episodes), desc=desc):
            p_obs_raw, e_obs_raw = env.reset()
            done = False
            steps = 0
            
            pursuer_agent.reset_episode()
            if mode == 'trained':
                evader_agent.reset_episode()
                
            # Track old distances for Δd_t calculation
            old_distances = [compute_nearest_distance(env, i) for i in range(env.n_evaders)]
            
            while not done and steps < env.max_steps:
                p_dicts = [obs_to_dict(o, env.grid_size) for o in p_obs_raw]
                p_actions = [pursuer_agent.select_action(obs, epsilon=0.0, use_target_encoder=True) for obs in p_dicts]
                
                if mode == 'trained':
                    e_dicts = [obs_to_dict(o, env.grid_size) for o in e_obs_raw]
                    e_actions = [evader_agent.select_action(obs, epsilon=0.0, use_target_encoder=True) for obs in e_dicts]
                else:
                    # The Random Walk Baseline cross-play substitute
                    e_actions = [np.random.choice(5) for _ in range(env.n_evaders)]
                
                # Step environment
                p_obs_raw, e_obs_raw, p_costs, e_costs, done = env.step(p_actions, e_actions)
                
                # Update histories if trained
                pursuer_agent.store_transition(p_dicts[0], p_actions[0], p_costs[0])
                if mode == 'trained':
                    evader_agent.store_transition(e_dicts[0], e_actions[0], e_costs[0])
                
                # Evaluate Distance Dynamics (Δd_t)
                new_distances = [compute_nearest_distance(env, i) for i in range(env.n_evaders)]
                for i in range(env.n_evaders):
                    d_old = old_distances[i]
                    d_new = new_distances[i]
                    # Only map causality if under immediate threat (Manhattan distance <= 4 is approx FOV)
                    if d_old <= 4:
                        delta = d_new - d_old
                        delta_distances.append(delta)
                        
                old_distances = new_distances
                steps += 1
            
            capture_history.append(1.0 if env.captured else 0.0)
            episode_lengths.append(steps)
            
    return capture_history, episode_lengths, delta_distances


def evaluate_crossplay_baseline(checkpoint_path, grid_size=25, n_pursuers=3, n_evaders=2, eval_episodes=200):
    print(f"Loading Crossplay Baseline Checkpoint: {checkpoint_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = PursuitEvasionEnv(grid_size=grid_size, n_pursuers=n_pursuers, n_evaders=n_evaders)
    pursuer_agent = BSRMCHFAAgent(grid_size=grid_size, device=device)
    evader_agent = BSRMCHFAAgent(grid_size=grid_size, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    pursuer_agent.history_encoder.load_state_dict(checkpoint['pursuer_encoder'])
    pursuer_agent.target_encoder.load_state_dict(checkpoint['pursuer_encoder'])
    pursuer_agent.belief_engine.load_state_dict(checkpoint['pursuer_belief'])
    pursuer_agent.regret_minimizer.load_state_dict(checkpoint['pursuer_regret'])
    
    evader_agent.history_encoder.load_state_dict(checkpoint['evader_encoder'])
    evader_agent.target_encoder.load_state_dict(checkpoint['evader_encoder'])
    evader_agent.belief_engine.load_state_dict(checkpoint['evader_belief'])
    evader_agent.regret_minimizer.load_state_dict(checkpoint['evader_regret'])
    
    for net in [pursuer_agent.history_encoder, pursuer_agent.target_encoder, 
                pursuer_agent.belief_engine, pursuer_agent.regret_minimizer,
                evader_agent.history_encoder, evader_agent.target_encoder, 
                evader_agent.belief_engine, evader_agent.regret_minimizer]:
        net.eval()
    
    print("\n--- Phase 1: Control (Trained vs Trained) ---")
    trained_cap, trained_len, trained_delta = run_evaluation_mode(env, pursuer_agent, evader_agent, 'trained', eval_episodes)
    
    print("\n--- Phase 2: Treatment (Trained vs Random) ---")
    random_cap, random_len, random_delta = run_evaluation_mode(env, pursuer_agent, evader_agent, 'random', eval_episodes)

    generate_crossplay_plots(trained_cap, random_cap, trained_delta, random_delta, eval_episodes)


def generate_crossplay_plots(trained_cap, random_cap, trained_delta, random_delta, eval_episodes):
    sns.set_theme(style="whitegrid")
    os.makedirs('plots', exist_ok=True)
    
    # 1. The Crossplay Proof P(T=100)
    plt.figure(figsize=(8, 6))
    
    cum_trained = np.cumsum(trained_cap) / np.arange(1, eval_episodes + 1) * 100
    cum_random = np.cumsum(random_cap) / np.arange(1, eval_episodes + 1) * 100
    
    plt.plot(cum_trained, color='#1f77b4', linewidth=3, label="Trained Evaders")
    plt.plot(cum_random, color='#ff7f0e', linewidth=3, linestyle='--', label="Random Evaders")
    
    plt.title("Crossplay Proof: Pursuit Dominance vs Evasion Intelligence", fontsize=15, fontweight='bold')
    plt.xlabel("Evaluation Episode", fontsize=13)
    plt.ylabel("Cumulative Capture Rate (%)", fontsize=13)
    plt.ylim(0, 100)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/crossplay_capture_rate.png', format='png', dpi=300)
    plt.close()
    
    # 2. Distance Dynamics Under Proximity E[Δd_t]
    plt.figure(figsize=(7, 6))
    
    # Filter out empty or NaN to compute clean means
    e_trained = np.mean(trained_delta) if trained_delta else 0.0
    e_random = np.mean(random_delta) if random_delta else 0.0
    
    sns.barplot(x=["Trained Evader", "Random Evader"], y=[e_trained, e_random], palette=['#1f77b4', '#ff7f0e'])
    
    plt.title(r"Distance Dynamics Under Proximity ($\mathbb{E}[\Delta d_t]$)", fontsize=15, fontweight='bold')
    plt.ylabel("Expected Manhattan Distance Change", fontsize=13)
    plt.axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('plots/crossplay_distance_dynamics.png', format='png', dpi=300)
    plt.close()
    
    print(f"\n==========================================")
    print(f"Crossplay Analysis Complete!")
    print(f"Metrics Generated:")
    print(f"  Trained Evader Capture Rate: {np.mean(trained_cap)*100:.1f}%")
    print(f"  Random Evader Capture Rate:  {np.mean(random_cap)*100:.1f}%")
    print(f"  Trained E[Δd_t] in FOV:      {e_trained:.3f} steps")
    print(f"  Random  E[Δd_t] in FOV:      {e_random:.3f} steps")
    print(f"Plots saved to 'plots/'")
    print(f"==========================================")

if __name__ == "__main__":
    checkpoint_files = glob.glob("checkpoints/checkpoint_v2_ep*.pt")
    if not checkpoint_files:
        print("No checkpoints found.")
    else:
        latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_ep')[1].split('.pt')[0]))[-1]
        # Evaluates 200 episodes for both groups dynamically
        evaluate_crossplay_baseline(latest_checkpoint, eval_episodes=200)
