"""
B-SRM-CHFA Agent: Belief-State Regret Minimization with Contextual History
and Function Approximation

Three core components:
1. Contextual History Encoder (LSTM) - compresses observation sequences
2. Belief Inference Engine (VAE) - maps history to belief distribution
3. Neural Regret Minimizer (MLP) - computes regrets for actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque
from typing import Dict, List, Tuple, Optional


class ContextualHistoryEncoder(nn.Module):
    """
    LSTM-based encoder that compresses unbounded observation-action-cost histories
    into fixed-size embeddings.
    
    Input: sequence of (observation, action, cost) tuples
    Output: 256-dim embedding z_t
    """
    
    def __init__(
        self,
        fov_size: int = 5,
        fov_channels: int = 3,
        action_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.fov_size = fov_size
        self.hidden_dim = hidden_dim
        
        # Observation encoder (CNN for FOV grid)
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(fov_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 64 * 5 * 5 = 1600
            nn.Linear(64 * fov_size * fov_size, 128)
        )
        
        # Position encoder
        self.pos_encoder = nn.Linear(2, 16)  # (x, y)
        
        # Last known opponent encoder
        self.opponent_encoder = nn.Linear(3, 16)  # (x, y, steps_ago)
        
        # Action encoder
        self.action_encoder = nn.Embedding(action_dim, 16)
        
        # Input to LSTM: obs(128) + pos(16) + opponent(16) + action(16) + cost(1) = 177
        lstm_input_dim = 128 + 16 + 16 + 16 + 1
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(
        self, 
        history: List[Dict], 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode observation history.
        
        Args:
            history: List of dicts with keys: 'obs', 'action', 'cost'
                Each obs dict has: 'fov_grid', 'own_pos', 'last_known_opponent'
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            embedding: (batch_size, hidden_dim) embedding z_t
            hidden: Updated LSTM hidden state
        """
        if len(history) == 0:
            # Initialize with zeros
            batch_size = 1
            device = next(self.parameters()).device
            embedding = torch.zeros(batch_size, self.hidden_dim, device=device)
            h = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
            c = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
            return embedding, (h, c)
        
        # Process sequence
        sequence_features = []
        
        for step in history:
            obs = step['obs']
            action = step['action']
            cost = step['cost']
            
            # Encode FOV grid (add batch dimension)
            fov = torch.FloatTensor(obs['fov_grid']).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            obs_enc = self.obs_encoder(fov)  # (1, 128)
            
            # Encode position
            pos = torch.FloatTensor(obs['own_pos']).unsqueeze(0) / 25.0  # Normalize to [0, 1]
            pos_enc = self.pos_encoder(pos)  # (1, 16)
            
            # Encode last known opponent
            opponent = torch.FloatTensor(obs['last_known_opponent']).unsqueeze(0)
            opponent[..., :2] /= 25.0  # Normalize positions
            opponent[..., 2] /= 150.0  # Normalize steps_ago
            opponent_enc = self.opponent_encoder(opponent)  # (1, 16)
            
            # Encode action
            action_tensor = torch.LongTensor([action])
            action_enc = self.action_encoder(action_tensor)  # (1, 16)
            
            # Cost
            cost_tensor = torch.FloatTensor([[cost]])  # (1, 1)
            
            # Concatenate
            step_features = torch.cat([obs_enc, pos_enc, opponent_enc, action_enc, cost_tensor], dim=1)
            sequence_features.append(step_features)
        
        # Stack into sequence (1, seq_len, feature_dim)
        sequence = torch.cat(sequence_features, dim=0).unsqueeze(0)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(sequence, hidden)
        
        # Return final hidden state as embedding
        embedding = lstm_out[:, -1, :]  # (1, hidden_dim)
        
        return embedding, hidden


class BeliefInferenceEngine(nn.Module):
    """
    VAE that maps history embedding to belief distribution over states.
    
    Input: history embedding z_t (256-dim)
    Output: belief distribution b_t over grid positions
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        grid_size: int = 25,
        latent_dim: int = 32
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        
        # Encoder: z_t -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: latent -> belief distribution
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size * grid_size)
        )
        
    def encode(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters"""
        h = self.encoder(z_t)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to belief distribution"""
        logits = self.decoder(z)
        # Reshape to (batch, grid_size, grid_size)
        logits = logits.view(-1, self.grid_size, self.grid_size)
        # Softmax to get valid probability distribution
        belief = F.softmax(logits.view(-1, self.grid_size * self.grid_size), dim=1)
        belief = belief.view(-1, self.grid_size, self.grid_size)
        return belief
    
    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            belief: (batch, grid_size, grid_size) probability distribution
            mu: latent mean
            logvar: latent log variance
        """
        mu, logvar = self.encode(z_t)
        z = self.reparameterize(mu, logvar)
        belief = self.decode(z)
        return belief, mu, logvar
    
    def get_belief(self, z_t: torch.Tensor) -> torch.Tensor:
        """Get belief without sampling (use mean)"""
        mu, _ = self.encode(z_t)
        belief = self.decode(mu)
        return belief


class NeuralRegretMinimizer(nn.Module):
    """
    MLP that computes regrets for each action given history embedding and belief.
    
    Input: (z_t, b_t_summary, action_onehot)
    Output: regret value R(I, a)
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        belief_summary_dim: int = 32,  # Summary statistics from belief
        action_dim: int = 5,
        hidden_dims: List[int] = [512, 256, 128]
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Belief summarizer (reduce 25x25 grid to compact representation)
        self.belief_summarizer = nn.Sequential(
            nn.Linear(25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, belief_summary_dim)
        )
        
        # Regret network
        input_dim = embedding_dim + belief_summary_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output: single regret value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.regret_net = nn.Sequential(*layers)
    
    def forward(
        self, 
        z_t: torch.Tensor, 
        belief: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regrets for actions.
        
        Args:
            z_t: (batch, embedding_dim) history embeddings
            belief: (batch, grid_size, grid_size) belief distributions
            actions: (batch, num_actions) one-hot encoded actions
        
        Returns:
            regrets: (batch, num_actions) regret values
        """
        batch_size = z_t.shape[0]
        
        # Summarize belief
        belief_flat = belief.view(batch_size, -1)
        belief_summary = self.belief_summarizer(belief_flat)  # (batch, belief_summary_dim)
        
        # Expand z_t and belief_summary for each action
        z_t_expanded = z_t.unsqueeze(1).repeat(1, self.action_dim, 1)  # (batch, num_actions, embed_dim)
        belief_expanded = belief_summary.unsqueeze(1).repeat(1, self.action_dim, 1)  # (batch, num_actions, belief_dim)
        
        # Create one-hot action vectors
        action_onehot = F.one_hot(torch.arange(self.action_dim), self.action_dim).float()
        action_onehot = action_onehot.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, num_actions, action_dim)
        
        # Concatenate features
        features = torch.cat([z_t_expanded, belief_expanded, action_onehot], dim=2)  # (batch, num_actions, total_dim)
        
        # Compute regrets
        regrets = self.regret_net(features).squeeze(-1)  # (batch, num_actions)
        
        return regrets


class OpponentPredictor(nn.Module):
    """Auxiliary task: predict opponent's next observation"""
    
    def __init__(self, embedding_dim: int = 256, grid_size: int = 25):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, grid_size * grid_size)
        )
        
        self.grid_size = grid_size
    
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict opponent position distribution"""
        logits = self.predictor(z_t)
        probs = F.softmax(logits, dim=1)
        return probs.view(-1, self.grid_size, self.grid_size)


class ValueNetwork(nn.Module):
    """Estimates cost-to-go V(s) for TD learning"""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Estimate value"""
        return self.value_net(z_t).squeeze(-1)


class ReservoirBuffer:
    """Reservoir sampling buffer for regret storage"""
    
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.total_added = 0
    
    def add(self, z_t, belief, action, regret):
        """Add experience with reservoir sampling"""
        experience = {
            'z_t': z_t.detach().cpu(),
            'belief': belief.detach().cpu(),
            'action': action,
            'regret': regret
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # Reservoir sampling
            self.position = np.random.randint(0, self.total_added + 1)
            if self.position < self.capacity:
                self.buffer[self.position] = experience
        
        self.total_added += 1
    
    def sample(self, batch_size: int) -> Dict:
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'z_t': torch.stack([b['z_t'] for b in batch]),
            'belief': torch.stack([b['belief'] for b in batch]),
            'action': torch.LongTensor([b['action'] for b in batch]),
            'regret': torch.FloatTensor([b['regret'] for b in batch])
        }
    
    def __len__(self):
        return len(self.buffer)


class BSRMCHFAAgent:
    """
    Complete B-SRM-CHFA agent combining all components.
    """
    
    def __init__(
        self,
        grid_size: int = 25,
        fov_size: int = 5,
        action_dim: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 3e-4,
        reservoir_capacity: int = 1_000_000
    ):
        self.device = device
        self.grid_size = grid_size
        self.action_dim = action_dim
        
        # Initialize networks
        self.history_encoder = ContextualHistoryEncoder(
            fov_size=fov_size, 
            hidden_dim=256
        ).to(device)
        
        # Target encoder for stable Phase 2 features
        self.target_encoder = copy.deepcopy(self.history_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        
        self.belief_engine = BeliefInferenceEngine(
            embedding_dim=256, 
            grid_size=grid_size
        ).to(device)
        
        self.regret_minimizer = NeuralRegretMinimizer(
            embedding_dim=256,
            action_dim=action_dim
        ).to(device)
        
        self.opponent_predictor = OpponentPredictor(
            embedding_dim=256,
            grid_size=grid_size
        ).to(device)
        
        self.value_network = ValueNetwork(embedding_dim=256).to(device)
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.history_encoder.parameters(), lr=learning_rate
        )
        self.belief_optimizer = torch.optim.Adam(
            self.belief_engine.parameters(), lr=learning_rate
        )
        self.regret_optimizer = torch.optim.Adam(
            self.regret_minimizer.parameters(), lr=learning_rate
        )
        self.opponent_optimizer = torch.optim.Adam(
            self.opponent_predictor.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=learning_rate
        )
        
        # Reservoir buffer
        self.reservoir = ReservoirBuffer(capacity=reservoir_capacity)
        
        # History tracking
        self.history = []
        self.lstm_hidden = None
        
    def soft_update_target_encoder(self, tau: float = 0.005):
        """Soft update target encoder weights: target = tau * live + (1 - tau) * target"""
        for target_param, live_param in zip(self.target_encoder.parameters(), self.history_encoder.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)
            
    def reset_episode(self):
        """Reset for new episode"""
        self.history = []
        self.lstm_hidden = None
    
    def select_action(self, obs: Dict, epsilon: float = 0.0, use_target_encoder: bool = True) -> int:
        """
        Select action using regret-matching+.
        
        Args:
            obs: Current observation
            epsilon: Exploration rate (for epsilon-greedy)
        
        Returns:
            action: Selected action (0-4)
        """
        # Encode history
        with torch.no_grad():
            encoder = self.target_encoder if use_target_encoder else self.history_encoder
            z_t, self.lstm_hidden = encoder(self.history, self.lstm_hidden)
            
            # Infer belief
            belief = self.belief_engine.get_belief(z_t)
            
            # Compute regrets
            regrets = self.regret_minimizer(z_t, belief, None)  # (1, action_dim)
            regrets = regrets.squeeze(0).cpu().numpy()
        
        # Regret-matching+
        positive_regrets = np.maximum(regrets, 0)
        
        if positive_regrets.sum() > 0:
            probs = positive_regrets / positive_regrets.sum()
        else:
            probs = np.ones(self.action_dim) / self.action_dim
        
        # Epsilon-greedy (for exploration)
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.random.choice(self.action_dim, p=probs)
        
        return action
    
    def store_transition(self, obs: Dict, action: int, cost: float):
        """Store transition in history"""
        self.history.append({
            'obs': obs,
            'action': action,
            'cost': cost
        })
    
    def compute_counterfactual_regret(
        self, 
        true_state: np.ndarray,
        action_taken: int,
        cost: float,
        z_t: torch.Tensor,
        belief: torch.Tensor
    ) -> float:
        """
        Compute counterfactual regret for the action taken.
        
        This is a simplified version - full implementation would use
        counterfactual value computation.
        """
        with torch.no_grad():
            # Compute value of current state
            value_current = self.value_network(z_t).item()
            
            # Regret = (expected cost under best action) - (actual cost)
            # Simplified: regret ≈ value_current - cost
            regret = value_current - cost
        
        return regret
    
    def update(
        self,
        batch_size: int = 256,
        belief_loss_weight: float = 1.0,
        opponent_loss_weight: float = 0.1
    ):
        """Update all networks"""
        if len(self.reservoir) < batch_size:
            return
        
        # Sample batch from reservoir
        batch = self.reservoir.sample(batch_size)
        
        z_t = batch['z_t'].to(self.device)
        belief = batch['belief'].to(self.device)
        actions = batch['action'].to(self.device)
        target_regrets = batch['regret'].to(self.device)
        
        # Update regret minimizer
        predicted_regrets = self.regret_minimizer(z_t, belief, actions)
        regret_loss = F.mse_loss(
            predicted_regrets.gather(1, actions.unsqueeze(1)).squeeze(1),
            target_regrets
        )
        
        self.regret_optimizer.zero_grad()
        regret_loss.backward()
        self.regret_optimizer.step()
        
        return {
            'regret_loss': regret_loss.item()
        }


if __name__ == '__main__':
    print("Testing B-SRM-CHFA Agent...")
    
    # Create agent
    agent = BSRMCHFAAgent(grid_size=25, fov_size=5)
    
    # Dummy observation
    obs = {
        'fov_grid': np.random.rand(5, 5, 3),
        'own_pos': np.array([10, 10]),
        'last_known_opponent': np.array([15, 15, 5])
    }
    
    # Reset episode
    agent.reset_episode()
    
    # Select action
    action = agent.select_action(obs)
    print(f"Selected action: {action}")
    
    # Store transition
    agent.store_transition(obs, action, cost=1.0)
    
    # Select another action
    action2 = agent.select_action(obs)
    print(f"Selected action 2: {action2}")
    
    print("\nAgent test completed successfully!")
