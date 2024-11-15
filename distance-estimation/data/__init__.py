"""Provides a way to read the dataset from disk."""

from collections import namedtuple
from typing import Optional

import h5py
import numpy as np
import torch

CPUTrainingSample = namedtuple(
    "CPUTrainingSample", ["source_idx", "target_idx", "distance"]
)
"""
Training example represented on the CPU-side. These examples have to be
post-processed on the GPU to be used in the model.
"""

GPUTrainingSample = namedtuple(
    "GPUTrainingSample", ["source", "target", "distance", "weight"]
)
"""A training example represented on the GPU-side, ready for the model."""


class DistanceEstimationDataset(torch.utils.data.Dataset):
    """
    Read the HDF5 dataset containing the distance estimation data.

    The HDF5 file has 4 datasets:
     * `bfs/source-idx` (T,): The source node indices.
     * `bfs/target-idx` (T, N-1): The target node indices.
     * `bfs/distance` (T, N-1): The distance between the source and target.
     * `edge/pairs` (K, 2): Edges to compensate for distance one.
    Here, N is the number of nodes in the graph.
    """

    def __init__(
        self,
        hdf5_filename: str,
        embedding_filename: str,
        embedding_length: int,
        max_distance: int,
        num_bfs: Optional[int] = None,
        num_edge: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the dataset.

        We can optionally truncate the dataset to a certain number of runs and
        edges. This is useful for debugging and testing. Also, we have to
        specify the number of categories as `max_distance`. It is exclusive.
        """

        # Handle the embeddings
        emb = np.load(embedding_filename)
        assert emb.ndim == 2
        assert emb.shape[1] >= embedding_length
        emb = emb[:, :embedding_length]
        self.embeddings = torch.from_numpy(emb).to(device)
        self.embeddings /= torch.linalg.vector_norm(
            self.embeddings, dim=1, keepdim=True
        )

        # Read the HDF5 file
        self.hdf5 = h5py.File(hdf5_filename, "r")
        self.bfs = self.hdf5["bfs"]
        self.edge = self.hdf5["edge"]
        assert len(self.bfs["source-idx"].shape) == 1
        assert len(self.bfs["target-idx"].shape) == 2
        assert len(self.bfs["distance"].shape) == 2
        assert len(self.edge["pairs"].shape) == 2
        assert self.bfs["source-idx"].shape[0] == self.bfs["distance"].shape[0]
        assert self.bfs["target-idx"].shape[0] == self.bfs["distance"].shape[0]
        assert self.bfs["target-idx"].shape[1] == self.bfs["distance"].shape[1]
        assert self.edge["pairs"].shape[1] == 2

        # Populate parameters about the dataset
        self.N = self.bfs["distance"].shape[1] + 1
        self.T = self.bfs["distance"].shape[0]
        self.K = self.edge["pairs"].shape[0]
        if num_bfs is not None and self.T >= num_bfs:
            self.T = num_bfs
        if num_edge is not None and self.K >= num_edge:
            self.K = num_edge

        # Compute class counts
        self.class_counts = np.zeros(max_distance, dtype=np.uint64)
        assert max_distance > 2
        for cnk in self.bfs["distance"].iter_chunks(
            sel=np.s_[0 : self.T, 0 : self.N - 1]
        ):
            arr = self.bfs["distance"][cnk]
            arr = np.where(arr >= max_distance, 0, arr)
            num, _ = np.histogram(arr, bins=np.arange(max_distance + 1))
            self.class_counts += num.astype(np.uint64)
        self.class_counts[1] += self.K
        # Compute class weights from counts
        weight_const = np.sum(self.class_counts) / max_distance
        self.class_weights = weight_const / self.class_counts
        self.max_distance = max_distance
        # Move everything to the device
        self.class_counts = torch.from_numpy(self.class_counts).to(device)
        self.class_weights = torch.from_numpy(self.class_weights).to(device)

        self.device = device

    def __len__(self):
        t_batches, k_batches = self._get_batchlen()
        return t_batches + k_batches

    def _get_batchlen(self) -> int:
        t_batches = self.T
        k_batches = self.K // (self.N - 1)
        k_batches += 1 if self.K % (self.N - 1) != 0 else 0
        return t_batches, k_batches

    def __getitem__(self, idx) -> CPUTrainingSample:
        """
        Return a single training sample.

        The source and target are left as indices. They will be "decompressed"
        on the GPU. The same is done for the distance - the sample weight is
        kept on the GPU too. Also, any data outside of the maximum distance will
        be mapped to 0.

        Finally, we make batches of size N-1. This really speeds up training,
        since otherwise we're CPU bound on reading the data.
        """
        t_batches, k_batches = self._get_batchlen()
        if idx < t_batches:
            return self._get_bfs(idx)
        elif idx < t_batches + k_batches:
            return self._get_edge(idx - t_batches)
        else:
            raise IndexError("Index out of range")

    def _get_bfs(self, i_T: int) -> CPUTrainingSample:
        s = torch.from_numpy(self.bfs["source-idx"][i_T : i_T + 1])
        t = torch.from_numpy(self.bfs["target-idx"][i_T, :])
        d = torch.from_numpy(self.bfs["distance"][i_T, :])

        s = s.expand(self.N - 1)
        d[d >= self.max_distance] = 0
        return CPUTrainingSample(s, t, d)

    def _get_edge(self, i_Kb: int) -> CPUTrainingSample:
        i_K0 = i_Kb * (self.N - 1)
        i_K1 = min(i_K0 + (self.N - 1), self.K)
        i_Klen = i_K1 - i_K0
        assert i_Klen > 0

        s = torch.from_numpy(self.edge["pairs"][i_K0:i_K1, 0])
        t = torch.from_numpy(self.edge["pairs"][i_K0:i_K1, 1])
        return CPUTrainingSample(s, t, torch.ones(i_Klen, dtype=torch.uint8))

    def process_batch(self, batch: CPUTrainingSample) -> GPUTrainingSample:
        """
        Expand the source and target indices, and get the sample weights.
        """
        s, t, d = batch
        s = s.view(-1)
        t = t.view(-1)
        d = d.view(-1)

        s = s.to(self.device).long()
        t = t.to(self.device).long()
        d = d.to(self.device).long()

        s = torch.index_select(self.embeddings, 0, s)
        t = torch.index_select(self.embeddings, 0, t)
        w = torch.index_select(self.class_weights, 0, d)

        return GPUTrainingSample(s, t, d, w)
