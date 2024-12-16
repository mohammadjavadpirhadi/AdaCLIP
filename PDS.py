import math
import random
from typing import Iterator, Optional, TypeVar, List

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


__all__ = ["PairedDistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


class PairedDistributedSampler(Sampler[_T_co]):

    def __init__(
        self,
        pos_indices: List[int],
        neg_indices: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if num_replicas % 2 != 0:
            raise ValueError("Only even number of world size is acceptable.")
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        self.bottleneck_length = min(len(pos_indices), len(neg_indices))

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and (self.bottleneck_length*2) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                ((self.bottleneck_length*2) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil((self.bottleneck_length*2) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            pos_indices = torch.randperm(len(self.pos_indices), generator=g).tolist()  # type: ignore[arg-type]
            neg_indices = torch.randperm(len(self.neg_indices), generator=g).tolist()  # type: ignore[arg-type]
        else:
            pos_indices = list(range(len(self.pos_indices)))  # type: ignore[arg-type]
            neg_indices = list(range(len(self.neg_indices)))  # type: ignore[arg-type]
        indices = []
        for i in range(self.bottleneck_length):
            current_pair = [pos_indices[i], neg_indices[i]]
            random.suffle(current_pair)
            indices.extend(current_pair)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
