"""
Belief Inference Engine
Implements f_φ(z_t) → b_t from B-SRM-CHFA paper (Section 3.3)

Maps contextual history embeddings to probability distributions over states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class BeliefInferenceEngine(nn.Module):
    """
    Maps history embedding z_t to belief distribution over states.
    
    For discrete state spaces (grid positions), outputs categorical distribution.
    Trained via supervised cross-entropy loss with true state during centralized training.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        grid_size: int = 10,
        n_agents: int = 5,  # Total agents to track (pursuers + evaders)
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        use_separate_agent_beliefs: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.use_separate_agent_beliefs = use_separate_agent_beliefs
        
        # State space: each agent has grid_size^2 possible positions
        # We predict a factored belief: P(s) = ∏_i P(s_i)
        # This assumes agent positions are conditionally independent given observations
        self.state_dim_per_agent = grid_size * grid_size
        
        # MLP layers
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads - one per agent
        if use_separate_agent_beliefs:
            self.belief_heads = nn.ModuleList([
                nn.Linear(prev_dim, self.state_dim_per_agent)
                for _ in range(n_agents)
            ])
        else:
            # Single head for joint belief (much larger output)
            self.belief_head = nn.Linear(prev_dim, n_agents * self.state_dim_per_agent)
    
    def forward(self, z_t: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Infer belief distribution from history embedding.
        
        Args:
            z_t: (batch, embedding_dim) contextual history embedding
            return_logits: if True, return raw logits (for training with cross_entropy).
                           if False, return softmax probabilities (for inference).
            
        Returns:
            belief: (batch, n_agents, grid_size^2)
                    Probabilities if return_logits=False, raw logits if True.
        """
        batch_size = z_t.size(0)
        
        # Encode
        features = self.encoder(z_t)
        
        # Predict beliefs
        if self.use_separate_agent_beliefs:
            agent_logits = []
            for head in self.belief_heads:
                logits = head(features)  # (batch, grid_size^2)
                agent_logits.append(logits)
            
            all_logits = torch.stack(agent_logits, dim=1)  # (batch, n_agents, grid_size^2)
        else:
            all_logits = self.belief_head(features)  # (batch, n_agents * grid_size^2)
            all_logits = all_logits.reshape(batch_size, self.n_agents, -1)
        
        if return_logits:
            return all_logits
        else:
            return F.softmax(all_logits, dim=-1)
    
    def compute_loss(self, logits: torch.Tensor, true_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for belief learning.
        
        IMPORTANT: `logits` must be RAW logits (from forward(return_logits=True)),
        NOT softmaxed probabilities. F.cross_entropy applies log_softmax internally.
        
        Args:
            logits: (batch, n_agents, grid_size^2) raw logits
            true_positions: (batch, n_agents, 2) true (x, y) positions
            
        Returns:
            loss: scalar cross-entropy loss
        """
        # Convert (x, y) positions to flat indices
        true_indices = (true_positions[..., 1] * self.grid_size + 
                       true_positions[..., 0]).long()  # (batch, n_agents)
        
        # Clamp indices to valid range
        true_indices = true_indices.clamp(0, self.state_dim_per_agent - 1)
        
        # Compute cross-entropy for each agent
        loss = 0.0
        for i in range(self.n_agents):
            loss += F.cross_entropy(
                logits[:, i, :],        # raw logits, NOT softmax
                true_indices[:, i],
                reduction='mean'
            )
        
        return loss / self.n_agents
    
    def get_expected_positions(self, beliefs: torch.Tensor) -> torch.Tensor:
        """
        Compute expected (x, y) positions from belief distributions.
        
        Args:
            beliefs: (batch, n_agents, grid_size^2)
            
        Returns:
            expected_pos: (batch, n_agents, 2) expected (x, y) positions
        """
        batch_size, n_agents, _ = beliefs.shape
        
        # Create position grid
        x_coords = torch.arange(self.grid_size, device=beliefs.device).float()
        y_coords = torch.arange(self.grid_size, device=beliefs.device).float()
        
        # Meshgrid: (grid_size, grid_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten: (grid_size^2, 2)
        position_grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        
        # Compute expected positions
        # beliefs: (batch, n_agents, grid_size^2)
        # position_grid: (grid_size^2, 2)
        # Result: (batch, n_agents, 2)
        expected_pos = torch.einsum('bns,sc->bnc', beliefs, position_grid)
        
        return expected_pos


class GoalConditionedValueNetwork(nn.Module):
    """
    Estimates cost-to-goal V_ξ(s) for SSP settings.
    
    Provides dense gradient signal: "how far am I from capturing/being captured?"
    Trained via TD learning during centralized training.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64)
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output: single scalar cost-to-goal estimate
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict cost-to-goal.
        
        Args:
            state: (batch, state_dim) state vector
            
        Returns:
            value: (batch, 1) estimated cost-to-goal
        """
        return self.network(state)
    
    def compute_td_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute TD(0) loss for cost-to-goal estimation.
        
        V(s) should predict: c + V(s') for non-terminal states
        
        Args:
            states: (batch, state_dim)
            next_states: (batch, state_dim)
            costs: (batch, 1) immediate costs
            dones: (batch, 1) terminal flags
            gamma: discount factor (1.0 for undiscounted SSP)
            
        Returns:
            loss: scalar MSE loss
        """
        # Predict current values
        current_values = self.forward(states)
        
        # Predict next values (no gradient through target)
        with torch.no_grad():
            next_values = self.forward(next_states)
            # Zero out next values for terminal states
            next_values = next_values * (1.0 - dones)
        
        # TD target: c + γ * V(s')
        td_targets = costs + gamma * next_values
        
        # MSE loss
        loss = F.mse_loss(current_values, td_targets)
        
        return loss


def positions_to_state_vector(
    pursuer_positions: np.ndarray,
    evader_positions: np.ndarray,
    grid_size: int,
    timestep: int,
    max_steps: int
) -> np.ndarray:
    """
    Convert agent positions to normalized state vector.
    
    Args:
        pursuer_positions: (n_pursuers, 2) array of (x, y) positions
        evader_positions: (n_evaders, 2) array of (x, y) positions
        grid_size: grid dimension
        timestep: current timestep
        max_steps: episode horizon
        
    Returns:
        state: (2*n_agents + 1,) normalized state vector
    """
    n_pursuers = len(pursuer_positions)
    n_evaders = len(evader_positions)
    
    state = np.zeros(2 * (n_pursuers + n_evaders) + 1, dtype=np.float32)
    
    # Normalize positions to [0, 1]
    for i, pos in enumerate(pursuer_positions):
        state[2*i] = pos[0] / grid_size
        state[2*i + 1] = pos[1] / grid_size
    
    offset = 2 * n_pursuers
    for i, pos in enumerate(evader_positions):
        state[offset + 2*i] = pos[0] / grid_size
        state[offset + 2*i + 1] = pos[1] / grid_size
    
    # Normalized timestep
    state[-1] = timestep / max_steps
    
    return state


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test Belief Inference Engine
    grid_size = 10
    n_agents = 5  # 3 pursuers + 2 evaders
    embedding_dim = 256
    
    belief_engine = BeliefInferenceEngine(
        embedding_dim=embedding_dim,
        grid_size=grid_size,
        n_agents=n_agents,
        use_separate_agent_beliefs=True
    ).to(device)
    
    print(f"Belief engine parameters: {sum(p.numel() for p in belief_engine.parameters()):,}")
    
    # Create dummy batch
    batch_size = 8
    z_t = torch.randn(batch_size, embedding_dim).to(device)
    
    # Forward pass
    beliefs = belief_engine(z_t)
    print(f"\nBelief output shape: {beliefs.shape}")
    print(f"Belief sums (should be ~1.0): {beliefs.sum(dim=-1)}")
    
    # Test loss computation
    true_positions = torch.randint(0, grid_size, (batch_size, n_agents, 2)).to(device)
    loss = belief_engine.compute_loss(beliefs, true_positions)
    print(f"Cross-entropy loss: {loss.item():.4f}")
    
    # Test expected position computation
    expected_pos = belief_engine.get_expected_positions(beliefs)
    print(f"Expected positions shape: {expected_pos.shape}")
    print(f"Sample expected position: {expected_pos[0, 0]}")
    
    # Test Goal-Conditioned Value Network
    print("\n" + "="*50)
    print("Testing Goal-Conditioned Value Network...")
    
    state_dim = 2 * n_agents + 1  # Position pairs + timestep
    value_net = GoalConditionedValueNetwork(state_dim).to(device)
    
    print(f"Value network parameters: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Create dummy states
    states = torch.randn(batch_size, state_dim).to(device)
    next_states = torch.randn(batch_size, state_dim).to(device)
    costs = torch.rand(batch_size, 1).to(device)
    dones = torch.zeros(batch_size, 1).to(device)
    
    # Forward pass
    values = value_net(states)
    print(f"\nValue predictions shape: {values.shape}")
    print(f"Sample values: {values[:3, 0]}")
    
    # Test TD loss
    td_loss = value_net.compute_td_loss(states, next_states, costs, dones)
    print(f"TD loss: {td_loss.item():.4f}")
    
    # Test state vector conversion
    print("\n" + "="*50)
    print("Testing state vector conversion...")
    
    pursuer_pos = np.array([[3, 4], [5, 6], [7, 8]])
    evader_pos = np.array([[1, 2], [9, 0]])
    
    state_vec = positions_to_state_vector(
        pursuer_pos, evader_pos, grid_size=10, timestep=25, max_steps=100
    )
    
    print(f"State vector shape: {state_vec.shape}")
    print(f"State vector: {state_vec}")
    print(f"Pursuer 0 position (normalized): ({state_vec[0]:.2f}, {state_vec[1]:.2f}) -> ({state_vec[0]*10:.0f}, {state_vec[1]*10:.0f})")
    print(f"Timestep (normalized): {state_vec[-1]:.2f} -> {state_vec[-1]*100:.0f}/100")
    
    print("\n✓ All belief engine tests passed!")
