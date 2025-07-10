"""
Distributed samplers for Polar-RTDETRv2.

This module provides distributed samplers for multi-GPU and multi-node training.
It includes:
- DistributedSampler: A sampler that restricts data loading to a subset of the dataset
- NodeDistributedSampler: A sampler that restricts data loading to a subset of the dataset
  and is optimized for multi-node distributed training
"""

import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import numpy as np
from typing import Iterator, List, Optional, TypeVar, Dict, Any


T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
    """
    Sampler that restricts data loading to a subset of the dataset.

    This is a modified version of PyTorch's DistributedSampler that ensures
    each process gets a different subset of the data during training.

    Args:
        dataset: Dataset used for sampling
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process within num_replicas
        shuffle: Whether to shuffle the indices
        seed: Random seed for reproducibility
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ) -> None:
        """
        Initialize DistributedSampler.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        
        # Determine the total size of the sampler across all processes
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        """
        Iterate through the indices of the dataset.
        """
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make the dataset evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subsample based on rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        """
        Return the number of samples in the sampler.
        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This ensures different shuffling order at each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch


class NodeDistributedSampler(Sampler[T_co]):
    """
    Sampler that restricts data loading to a subset of the dataset,
    optimized for multi-node distributed training.

    This sampler ensures that each node gets a different subset of the data,
    and then further divides that subset among the GPUs within the node.

    Args:
        dataset: Dataset used for sampling
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process within num_replicas
        local_rank: Rank of the current process within the local node
        local_size: Number of processes in the local node
        shuffle: Whether to shuffle the indices
        seed: Random seed for reproducibility
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        local_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ) -> None:
        """
        Initialize NodeDistributedSampler.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            rank = dist.get_rank()
        
        # Determine local rank and size if not provided
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        
        # Calculate node rank and number of nodes
        node_rank = rank // local_size
        num_nodes = num_replicas // local_size
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.local_rank = local_rank
        self.local_size = local_size
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.epoch = 0
        self.drop_last = drop_last
        
        # Calculate samples per node
        samples_per_node = math.ceil(len(self.dataset) / self.num_nodes)
        self.samples_per_node = samples_per_node
        
        # Calculate samples per GPU within the node
        if self.drop_last and samples_per_node % self.local_size != 0:
            self.num_samples = math.ceil(
                (samples_per_node - self.local_size) / self.local_size
            )
        else:
            self.num_samples = math.ceil(samples_per_node / self.local_size)
        
        # Total size across all processes
        self.total_size = self.num_samples * self.local_size * self.num_nodes
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        """
        Iterate through the indices of the dataset.
        """
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make the dataset evenly divisible by num_nodes
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # First divide indices by node
        node_indices = indices[self.node_rank:self.total_size:self.num_nodes]
        
        # Then divide indices within the node by local rank
        local_indices = node_indices[self.local_rank:len(node_indices):self.local_size]
        
        assert len(local_indices) == self.num_samples

        return iter(local_indices)

    def __len__(self) -> int:
        """
        Return the number of samples in the sampler.
        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This ensures different shuffling order at each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
