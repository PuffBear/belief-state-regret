"""
Neural Regret Minimizer
Implements r_θ(z_t, a) → R̂(z_t, a) from B-SRM-CHFA paper (Section 3.4)

4-layer MLP with LayerNorm that maps (history embedding, action) pairs
to cumulative regret estimates. Regret-Matching+ converts these to action
probabilities for policy execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NeuralRegretMinimizer(nn.Module):
    """
    Maps (z_t, action) → cumulative regret estimate R̂(z_t, a).

    Architecture: 4-layer MLP with LayerNorm + ReLU + Dropout
    Input: concat(z_t ∈ ℝ^{embedding_dim}, onehot(a) ∈ ℝ^{n_actions})
    Output: scalar regret value
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        n_actions: int = 5,
        hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions

        input_dim = embedding_dim + n_actions

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # no dropout on last hidden layer
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer: scalar regret
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Xavier initialization for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute regret estimate for (embedding, action) pairs.

        Args:
            z_t: (batch, embedding_dim) history embeddings
            action: (batch,) action indices  OR  (batch, n_actions) one-hot

        Returns:
            regret: (batch, 1) cumulative regret estimates
        """
        # One-hot encode actions if needed
        if action.dim() == 1 or (action.dim() == 2 and action.shape[-1] != self.n_actions):
            if action.dim() == 2:
                action = action.squeeze(-1)
            action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
        else:
            action_onehot = action.float()

        x = torch.cat([z_t, action_onehot], dim=-1)
        return self.network(x)

    def get_all_regrets(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Compute regret estimates for ALL actions at once.

        Args:
            z_t: (batch, embedding_dim) history embeddings

        Returns:
            regrets: (batch, n_actions) regret for each action
        """
        batch_size = z_t.size(0)
        regrets = []
        for a in range(self.n_actions):
            action = torch.full((batch_size,), a, dtype=torch.long, device=z_t.device)
            r = self.forward(z_t, action)  # (batch, 1)
            regrets.append(r)
        return torch.cat(regrets, dim=-1)  # (batch, n_actions)

    def get_regret_matching_policy(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Convert regret estimates to action probabilities via Regret-Matching+.

        π(a | z_t) = max(0, R̂(z_t, a)) / Σ_a' max(0, R̂(z_t, a'))
        If all regrets ≤ 0, use uniform distribution.

        Args:
            z_t: (batch, embedding_dim)

        Returns:
            policy: (batch, n_actions) action probabilities
        """
        regrets = self.get_all_regrets(z_t)  # (batch, n_actions)

        # Regret-Matching+: clamp negatives to 0
        positive_regrets = torch.clamp(regrets, min=0.0)

        # Normalize
        regret_sum = positive_regrets.sum(dim=-1, keepdim=True)

        # If all regrets are zero/negative → uniform
        uniform = torch.ones_like(positive_regrets) / self.n_actions
        has_positive = (regret_sum > 1e-8).float()

        policy = has_positive * (positive_regrets / (regret_sum + 1e-8)) + \
                 (1.0 - has_positive) * uniform

        return policy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = NeuralRegretMinimizer(embedding_dim=256, n_actions=5).to(device)
    print(f"Regret network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Test forward pass
    batch = 8
    z_t = torch.randn(batch, 256, device=device)
    action = torch.randint(0, 5, (batch,), device=device)

    regret = net(z_t, action)
    print(f"Single-action regret shape: {regret.shape}")

    all_regrets = net.get_all_regrets(z_t)
    print(f"All-action regrets shape:   {all_regrets.shape}")
    print(f"Sample regrets: {all_regrets[0].detach().cpu().numpy()}")

    policy = net.get_regret_matching_policy(z_t)
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sums (should be 1.0): {policy.sum(dim=-1)}")
    print(f"Sample policy: {policy[0].detach().cpu().numpy()}")

    print("\n✓ Regret network test passed!")
