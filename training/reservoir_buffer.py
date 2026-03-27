"""
Reservoir Buffer for Regret Sampling
Implements reservoir sampling (Algorithm R) for B-SRM-CHFA training.

Stores (z_t, action, instantaneous_regret) tuples. Uniform sampling over
entire training history ensures no recency bias — all experiences are
equally likely to appear in training mini-batches.
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional


class ReservoirBuffer:
    """
    Fixed-capacity buffer with reservoir sampling.

    As new samples arrive beyond capacity, each new sample replaces a
    random existing sample with probability capacity/num_seen — guaranteeing
    uniform coverage of the full training history.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.num_seen = 0

        # Pre-allocate storage as lists (append then convert)
        self.embeddings: list = []        # z_t tensors
        self.actions: list = []           # action indices
        self.regrets: list = []           # instantaneous regret values
        self._size = 0

    def add(
        self,
        z_t: torch.Tensor,
        action: int,
        regret: float,
    ):
        """
        Add one (z_t, action, regret) tuple via reservoir sampling.

        Args:
            z_t: (embedding_dim,) detached history embedding
            action: integer action index
            regret: instantaneous regret scalar
        """
        z_t = z_t.detach().cpu()
        self.num_seen += 1

        if self._size < self.capacity:
            self.embeddings.append(z_t)
            self.actions.append(action)
            self.regrets.append(regret)
            self._size += 1
        else:
            # Replace with probability capacity / num_seen
            idx = random.randint(0, self.num_seen - 1)
            if idx < self.capacity:
                self.embeddings[idx] = z_t
                self.actions[idx] = action
                self.regrets[idx] = regret

    def add_batch(
        self,
        z_ts: torch.Tensor,
        actions: torch.Tensor,
        regrets: torch.Tensor,
    ):
        """
        Add a batch of tuples.

        Args:
            z_ts: (N, embedding_dim)
            actions: (N,) action indices
            regrets: (N,) regret values
        """
        z_ts = z_ts.detach().cpu()
        actions_np = actions.detach().cpu()
        regrets_np = regrets.detach().cpu()

        for i in range(z_ts.size(0)):
            self.add(z_ts[i], int(actions_np[i].item()), float(regrets_np[i].item()))

    def sample_batch(
        self,
        batch_size: int = 512,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a uniform random batch from the buffer.

        Args:
            batch_size: number of samples
            device: target device for returned tensors

        Returns:
            z_batch:      (batch_size, embedding_dim)
            action_batch: (batch_size,)
            regret_batch: (batch_size,)
        """
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty buffer")

        actual_batch = min(batch_size, self._size)
        indices = random.sample(range(self._size), actual_batch)

        z_batch = torch.stack([self.embeddings[i] for i in indices])
        action_batch = torch.tensor(
            [self.actions[i] for i in indices], dtype=torch.long
        )
        regret_batch = torch.tensor(
            [self.regrets[i] for i in indices], dtype=torch.float32
        )

        if device is not None:
            z_batch = z_batch.to(device)
            action_batch = action_batch.to(device)
            regret_batch = regret_batch.to(device)

        return z_batch, action_batch, regret_batch

    def __len__(self):
        return self._size

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has enough samples for a meaningful batch."""
        return self._size >= 64


if __name__ == "__main__":
    print("Testing ReservoirBuffer...")

    buf = ReservoirBuffer(capacity=1000)

    # Add 2000 samples (overflow capacity)
    for i in range(2000):
        z = torch.randn(256)
        buf.add(z, action=i % 5, regret=float(i) * 0.01)

    print(f"Buffer size: {len(buf)} (capacity: {buf.capacity})")
    print(f"Total seen: {buf.num_seen}")

    z_b, a_b, r_b = buf.sample_batch(32)
    print(f"Batch shapes: z={z_b.shape}, a={a_b.shape}, r={r_b.shape}")
    print(f"Action distribution: {[int((a_b == i).sum()) for i in range(5)]}")

    # Test batch add
    z_batch = torch.randn(64, 256)
    a_batch = torch.randint(0, 5, (64,))
    r_batch = torch.randn(64)
    buf.add_batch(z_batch, a_batch, r_batch)
    print(f"After batch add — size: {len(buf)}, seen: {buf.num_seen}")

    print("\n✓ ReservoirBuffer test passed!")
