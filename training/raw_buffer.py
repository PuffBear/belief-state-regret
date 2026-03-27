"""
Buffers for two-phase B-SRM-CHFA training.

RawExperienceBuffer: Stores raw observation histories + CTDE ground truth
                     for Phase 1 (encoder + belief + opponent training).

ValueBuffer: Stores (z_t, MC return) pairs for value network training.
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, List


class RawExperienceBuffer:
    """Stores raw episode data for Phase 1 training (representation learning).

    Each entry contains the full observation sequence + CTDE ground truth,
    enabling re-encoding with gradients for representation learning.

    Episode data format:
        obs_dicts:                List[dict]   - raw observation dicts per timestep
        actions:                  List[int]    - actions taken
        costs:                    List[float]  - costs received
        true_opponent_positions:  List[List[tuple]] - ground truth opponent positions
        opponent_actions:         List[int]    - actual opponent actions
    """

    def __init__(self, capacity: int = 200):
        self.buffer = deque(maxlen=capacity)

    def add(self, episode_data: Dict):
        self.buffer.append(episode_data)

    def sample(self, n: int) -> List[Dict]:
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class ValueBuffer:
    """Simple circular buffer for (z_t, mc_return) pairs."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, z_t: torch.Tensor, mc_return: float):
        self.buffer.append((z_t.detach().cpu(), mc_return))

    def sample(self, batch_size: int, device: str):
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        z = torch.stack([b[0].squeeze(0) for b in batch]).to(device)
        returns = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
        return z, returns

    def __len__(self):
        return len(self.buffer)
