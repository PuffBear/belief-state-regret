"""
Contextual History Encoder Module
Implements φ_ψ(h_t) from B-SRM-CHFA paper (Section 3.2)

Compresses unbounded observation-action-cost sequences into fixed-size embeddings.
Includes opponent modeling auxiliary task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from environments.pursuit_evasion_env import Observation, Action


class ObservationEmbedder(nn.Module):
    """
    Embeds local 5x5x3 grid observations into fixed-size vectors.
    Uses small CNN to process spatial structure.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int] = (5, 5, 3), embed_dim: int = 128):
        super().__init__()
        self.obs_shape = obs_shape
        self.embed_dim = embed_dim
        
        # Small CNN for spatial processing
        # Input: (batch, 3, 5, 5) - channels first for PyTorch
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Compute flattened size after convolutions
        # 5x5 grid with 32 channels = 800
        conv_out_size = obs_shape[0] * obs_shape[1] * 32
        
        self.fc = nn.Linear(conv_out_size, embed_dim)
        
    def forward(self, obs_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_grid: (batch, height, width, channels) or (batch, seq_len, h, w, c)
            
        Returns:
            embedded: (batch, embed_dim) or (batch, seq_len, embed_dim)
        """
        # Handle both batched and sequence inputs
        if obs_grid.dim() == 5:  # (batch, seq_len, h, w, c)
            batch_size, seq_len = obs_grid.shape[:2]
            # Flatten batch and sequence dimensions
            obs_grid = obs_grid.reshape(-1, *obs_grid.shape[2:])
            is_sequential = True
        else:
            is_sequential = False
        
        # Convert to channels-first: (batch, c, h, w)
        x = obs_grid.permute(0, 3, 1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Embed
        x = F.relu(self.fc(x))
        
        # Restore sequence dimension if needed
        if is_sequential:
            x = x.reshape(batch_size, seq_len, -1)
        
        return x


class ContextualHistoryEncoder(nn.Module):
    """
    LSTM-based encoder that processes observation-action-cost sequences.
    
    Input: h_t = [(o_0, a_0, c_0), (o_1, a_1, c_1), ..., (o_t, a_t, c_t)]
    Output: z_t ∈ R^d (contextual embedding)
    
    Includes opponent modeling auxiliary head for tracking opponent behavior.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (5, 5, 3),
        n_actions: int = 5,
        obs_embed_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        opponent_modeling: bool = True
    ):
        super().__init__()
        
        self.obs_embed_dim = obs_embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.opponent_modeling = opponent_modeling
        
        # Observation embedder
        self.obs_embedder = ObservationEmbedder(obs_shape, obs_embed_dim)
        
        # LSTM input: [obs_embed, action_onehot, cost]
        lstm_input_dim = obs_embed_dim + n_actions + 1
        
        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0 if n_layers == 1 else 0.1
        )
        
        # Opponent modeling head (predicts opponent's next observation)
        if opponent_modeling:
            # Predict discretized opponent position change (9 classes: 8 directions + stay)
            self.opponent_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 9)  # 9 possible opponent movement directions
            )
    
    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        cost_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Process a sequence of observations, actions, and costs.
        
        Args:
            obs_sequence: (batch, seq_len, h, w, c) observation grids
            action_sequence: (batch, seq_len) action indices
            cost_sequence: (batch, seq_len, 1) costs
            hidden_state: Optional previous LSTM hidden state
            
        Returns:
            z_t: (batch, hidden_dim) final contextual embedding
            new_hidden: Updated LSTM hidden state
            opponent_pred: (batch, 9) opponent movement predictions (if enabled)
        """
        batch_size, seq_len = obs_sequence.shape[:2]
        
        # Embed observations
        obs_embed = self.obs_embedder(obs_sequence)  # (batch, seq_len, obs_embed_dim)
        
        # One-hot encode actions
        action_onehot = F.one_hot(action_sequence, num_classes=self.n_actions).float()
        
        # Concatenate inputs
        lstm_input = torch.cat([obs_embed, action_onehot, cost_sequence], dim=-1)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)
        
        # Extract final hidden state as contextual embedding
        z_t = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Opponent modeling prediction (optional)
        opponent_pred = None
        if self.opponent_modeling:
            opponent_pred = self.opponent_head(z_t)  # (batch, 9)
        
        return z_t, new_hidden, opponent_pred
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state"""
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


class HistoryBuffer:
    """
    Stores observation-action-cost histories for online training.
    Handles variable-length sequences and batching.
    """
    
    def __init__(self, max_history_len: int = 100):
        self.max_history_len = max_history_len
        self.reset()
    
    def reset(self):
        """Clear history"""
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.costs: List[float] = []
        self.opponent_movements: List[int] = []  # For training opponent modeling
    
    def add(
        self,
        observation: Observation,
        action: int,
        cost: float,
        opponent_movement: Optional[int] = None
    ):
        """Add one timestep to history"""
        # Convert observation to numpy array
        obs_array = observation.local_grid  # (5, 5, 3)
        
        self.observations.append(obs_array)
        self.actions.append(action)
        self.costs.append(cost)
        
        if opponent_movement is not None:
            self.opponent_movements.append(opponent_movement)
        
        # Truncate if exceeds max length
        if len(self.observations) > self.max_history_len:
            self.observations.pop(0)
            self.actions.pop(0)
            self.costs.pop(0)
            if self.opponent_movements:
                self.opponent_movements.pop(0)
    
    def get_sequence(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current history as numpy arrays.
        
        Returns:
            obs_seq: (seq_len, h, w, c)
            action_seq: (seq_len,)
            cost_seq: (seq_len, 1)
        """
        obs_seq = np.stack(self.observations, axis=0)
        action_seq = np.array(self.actions, dtype=np.int64)
        cost_seq = np.array(self.costs, dtype=np.float32).reshape(-1, 1)
        
        return obs_seq, action_seq, cost_seq
    
    def __len__(self):
        return len(self.observations)


def compute_opponent_movement_label(
    prev_opponent_positions: List[Tuple[int, int]],
    curr_opponent_positions: List[Tuple[int, int]]
) -> int:
    """
    Compute discretized opponent movement direction.
    
    Returns:
        Label in {0, 1, ..., 8} representing:
        0: N, 1: NE, 2: E, 3: SE, 4: S, 5: SW, 6: W, 7: NW, 8: STAY
    """
    if not prev_opponent_positions or not curr_opponent_positions:
        return 8  # STAY (no detection)
    
    # Take first detected opponent (simplification)
    prev_pos = prev_opponent_positions[0]
    curr_pos = curr_opponent_positions[0]
    
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    
    # Map (dx, dy) to direction
    direction_map = {
        (0, -1): 0,   # N
        (1, -1): 1,   # NE
        (1, 0): 2,    # E
        (1, 1): 3,    # SE
        (0, 1): 4,    # S
        (-1, 1): 5,   # SW
        (-1, 0): 6,   # W
        (-1, -1): 7,  # NW
        (0, 0): 8     # STAY
    }
    
    # Clamp to valid movements
    dx = max(-1, min(1, dx))
    dy = max(-1, min(1, dy))
    
    return direction_map.get((dx, dy), 8)


if __name__ == "__main__":
    # Test encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder = ContextualHistoryEncoder(
        obs_shape=(5, 5, 3),
        n_actions=5,
        hidden_dim=256,
        opponent_modeling=True
    ).to(device)
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Create dummy batch
    batch_size = 4
    seq_len = 10
    
    obs_seq = torch.randn(batch_size, seq_len, 5, 5, 3).to(device)
    action_seq = torch.randint(0, 5, (batch_size, seq_len)).to(device)
    cost_seq = torch.randn(batch_size, seq_len, 1).to(device)
    
    # Forward pass
    hidden = encoder.init_hidden(batch_size, device)
    z_t, new_hidden, opp_pred = encoder(obs_seq, action_seq, cost_seq, hidden)
    
    print(f"\nInput shapes:")
    print(f"  Observations: {obs_seq.shape}")
    print(f"  Actions: {action_seq.shape}")
    print(f"  Costs: {cost_seq.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Contextual embedding z_t: {z_t.shape}")
    print(f"  Opponent prediction: {opp_pred.shape}")
    print(f"  Hidden state: h={new_hidden[0].shape}, c={new_hidden[1].shape}")
    
    # Test HistoryBuffer
    print("\n" + "="*50)
    print("Testing HistoryBuffer...")
    
    from pursuit_evasion_env import PursuitEvasionEnv
    
    env = PursuitEvasionEnv(grid_size=10)
    buffer = HistoryBuffer()
    
    pursuer_obs, evader_obs = env.reset()
    
    # Simulate 5 timesteps
    for t in range(5):
        p_actions = [Action.NORTH] * env.n_pursuers
        e_actions = [Action.SOUTH] * env.n_evaders
        
        p_obs, e_obs, p_costs, e_costs, done = env.step(p_actions, e_actions)
        
        # Add to buffer (for first pursuer)
        buffer.add(p_obs[0], p_actions[0].value, p_costs[0])
    
    obs_seq, action_seq, cost_seq = buffer.get_sequence()
    print(f"Buffer length: {len(buffer)}")
    print(f"Sequence shapes: obs={obs_seq.shape}, actions={action_seq.shape}, costs={cost_seq.shape}")
    
    # Convert to torch and encode
    obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device).float()
    action_tensor = torch.from_numpy(action_seq).unsqueeze(0).to(device)
    cost_tensor = torch.from_numpy(cost_seq).unsqueeze(0).to(device)
    
    hidden = encoder.init_hidden(1, device)
    z_t, _, opp_pred = encoder(obs_tensor, action_tensor, cost_tensor, hidden)
    
    print(f"Encoded history: z_t.shape = {z_t.shape}")
    print("✓ Encoder test passed!")
