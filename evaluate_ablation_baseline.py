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

from environments.pursuit_evasion_env import PursuitEvasionEnv
from agents.bsrmchfa_agent import BSRMCHFAAgent
from training.self_play_train import obs_to_dict

def evaluate_ablation_baseline(checkpoint_path, grid_size=25, n_pursuers=3, n_evaders=2, eval_episodes=500):
    print(f"Loading Ablation Baseline Checkpoint: {checkpoint_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize Environment and Agents
    env = PursuitEvasionEnv(grid_size=grid_size, n_pursuers=n_pursuers, n_evaders=n_evaders)
    
    pursuer_agent = BSRMCHFAAgent(grid_size=grid_size, device=device)
    evader_agent = BSRMCHFAAgent(grid_size=grid_size, device=device)
    
    # 2. Load the trained weights with strict=False to bypass missing Value Networks from old architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load each network individually
    pursuer_agent.history_encoder.load_state_dict(checkpoint['pursuer_encoder'], strict=False)
    pursuer_agent.target_encoder.load_state_dict(checkpoint['pursuer_encoder'], strict=False)
    pursuer_agent.belief_engine.load_state_dict(checkpoint['pursuer_belief'], strict=False)
    pursuer_agent.regret_minimizer.load_state_dict(checkpoint['pursuer_regret'], strict=False)
    
    evader_agent.history_encoder.load_state_dict(checkpoint['evader_encoder'], strict=False)
    evader_agent.target_encoder.load_state_dict(checkpoint['evader_encoder'], strict=False)
    evader_agent.belief_engine.load_state_dict(checkpoint['evader_belief'], strict=False)
    evader_agent.regret_minimizer.load_state_dict(checkpoint['evader_regret'], strict=False)
    
    # 3. CRITICAL: Lock the networks into Evaluation Mode
    for net in [pursuer_agent.history_encoder, pursuer_agent.target_encoder, 
                pursuer_agent.belief_engine, pursuer_agent.regret_minimizer,
                evader_agent.history_encoder, evader_agent.target_encoder, 
                evader_agent.belief_engine, evader_agent.regret_minimizer]:
        net.eval()
    
    # Metrics tracking
    capture_history = []
    episode_lengths = []
    
    print(f"\nEvaluating Ablation Average Strategy for {eval_episodes} Episodes...")
    
    # 4. The Evaluation Loop
    with torch.no_grad(): # No gradients during evaluation
        for episode in tqdm(range(eval_episodes)):
            p_obs_raw, e_obs_raw = env.reset()
            done = False
            steps = 0
            
            # Reset agent histories for the new episode
            pursuer_agent.reset_episode()
            evader_agent.reset_episode()
            
            while not done and steps < env.max_steps:
                p_dicts = [obs_to_dict(o, grid_size) for o in p_obs_raw]
                e_dicts = [obs_to_dict(o, grid_size) for o in e_obs_raw]

                p_actions = [pursuer_agent.select_action(obs, epsilon=0.0, use_target_encoder=True) for obs in p_dicts]
                e_actions = [evader_agent.select_action(obs, epsilon=0.0, use_target_encoder=True) for obs in e_dicts]
                
                # Step environment
                p_obs_raw, e_obs_raw, p_costs, e_costs, done = env.step(p_actions, e_actions)
                
                # Update histories
                pursuer_agent.store_transition(p_dicts[0], p_actions[0], p_costs[0])
                evader_agent.store_transition(e_dicts[0], e_actions[0], e_costs[0])
                
                steps += 1
            
            # Record metrics
            captured = env.captured
            capture_history.append(1.0 if captured else 0.0)
            episode_lengths.append(steps)

    # 5. Generate the NeurIPS-ready Plots
    generate_ablation_plots(capture_history, episode_lengths, eval_episodes)
    
    # Calculate final value of the game
    nash_value = np.mean(capture_history) * 100
    print(f"\n==========================================")
    print(f"Ablation Evaluation Complete!")
    print(f"Empirical Value of the Game (Frozen Representation Baseline): {nash_value:.2f}% Capture Rate")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"==========================================")


def generate_ablation_plots(capture_history, episode_lengths, eval_episodes):
    """Generates the stabilized evaluation charts for the paper."""
    sns.set_theme(style="darkgrid")
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: The Cumulative Capture Rate (The Nash Convergence Line)
    plt.figure(figsize=(7, 5))
    cumulative_capture = np.cumsum(capture_history) / np.arange(1, eval_episodes + 1)
    
    plt.plot(cumulative_capture * 100, color='#d62728', linewidth=2.5) # Using red for ablation
    plt.title(r"(Ablation) Average Strategy: $\epsilon$-Nash Equilibrium Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Evaluation Episode", fontsize=12)
    plt.ylabel("Cumulative Capture Rate (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('plots/ablation_nash_convergence.png', format='png', dpi=300)
    plt.close()
    
    # Plot 2: Episode Length Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(episode_lengths, bins=20, kde=True, color='#9467bd')
    plt.title("(Ablation) Distribution of Episode Horizons", fontsize=14, fontweight='bold')
    plt.xlabel("Steps to Termination", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/ablation_episode_horizons.png', format='png', dpi=300)
    plt.close()
    
    print("\nSaved separate high-res PNG plots to 'plots/ablation_nash_convergence.png' and 'plots/ablation_episode_horizons.png'")


if __name__ == "__main__":
    evaluate_ablation_baseline("checkpoints/checkpoint_episode_7000.pt", eval_episodes=500)
