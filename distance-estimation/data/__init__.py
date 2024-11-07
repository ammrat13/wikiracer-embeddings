"""Provides a way to read the dataset from disk."""

from typing import Optional

import h5py
import numpy as np
import torch


class DistanceEstimationDataset(torch.utils.data.Dataset):
    """
    Read the HDF5 dataset containing the distance estimation data.

    The HDF5 file has 3 datasets:
     * `source_idx` (T,): The source node indices.
     * `target_idx` (T, N-1): The target node indices.
     * `distance` (T, N-1): The distance between the source and target nodes.
    Here, T is the number of samples and N is the number of nodes in the graph.
    """

    TrainingInput = tuple[torch.Tensor, torch.Tensor]
    TrainingSample = tuple[TrainingInput, torch.Tensor, torch.Tensor]

    def __init__(
        self,
        hdf5_filename: str,
        embedding_filename: str,
        max_distance: int = 5,
        num_runs: Optional[int] = None,
        embedding_length: int = 512,
    ):
        """
        Initialize the dataset.

        We also have to take the array of embeddings. We essentially do
        dictionary coding on the embeddings. We also need to know how long the
        embeddings should be so we can truncate them.

        Additionally, we can optionally truncate the dataset to a certain number
        of "runs" (i.e. T). This way, we can characterize the model's
        performance on smaller datasets.

        Finally, we have to set a maximum distance before we just declare nodes
        unconnected.
        """

        # Read the Numpy array
        emb = np.load(embedding_filename)
        assert emb.ndim == 2
        assert emb.shape[1] >= embedding_length
        emb = emb[:, :embedding_length]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.embeddings = torch.from_numpy(emb)
        self.embedding_length = embedding_length

        # Read the HDF5 file
        self.hdf5_file = h5py.File(hdf5_filename, "r")
        assert len(self.hdf5_file["source-idx"].shape) == 1
        assert len(self.hdf5_file["target-idx"].shape) == 2
        assert len(self.hdf5_file["distance"].shape) == 2
        assert (
            self.hdf5_file["source-idx"].shape[0]
            == self.hdf5_file["target-idx"].shape[0]
        )
        assert (
            self.hdf5_file["target-idx"].shape[0] == self.hdf5_file["distance"].shape[0]
        )
        assert (
            self.hdf5_file["target-idx"].shape[1] == self.hdf5_file["distance"].shape[1]
        )

        # Populate parameters about the dataset
        self.T = self.hdf5_file["distance"].shape[0]
        self.N = self.hdf5_file["distance"].shape[1] + 1
        if num_runs is not None and self.T >= num_runs:
            self.T = num_runs

        # Do weighting
        counts = np.zeros(max_distance, dtype=np.uint64)
        for chunk in self.hdf5_file["distance"].iter_chunks():
            arr = self.hdf5_file["distance"][chunk]
            arr = np.where(arr >= max_distance, 0, arr)
            num, _ = np.histogram(arr, bins=np.arange(max_distance + 1))
            counts += num.astype(np.uint64)
        self.max_distance = max_distance
        self.class_weights = torch.from_numpy(np.sum(counts) / counts)

    def __len__(self):
        return self.T

    def __getitem__(self, idx) -> TrainingSample:
        """
        Get the idx-th sample from the dataset.

        Note that one sample coresponds to one source node and N-1 target nodes.
        These will also be batched, so you have to flatten.
        """

        s_idx = self.hdf5_file["source-idx"][idx]
        s = self.embeddings[s_idx].expand(self.N - 1, self.embedding_length)

        t_idx = self.hdf5_file["target-idx"][idx]
        t = torch.index_select(self.embeddings, 0, torch.from_numpy(t_idx).long())

        d = self.hdf5_file["distance"][idx]
        d = np.where(d >= self.max_distance, 0, d)
        d = torch.from_numpy(d).long()

        w_idx = torch.where(d < self.max_distance, d, 0)
        w = torch.index_select(self.class_weights, 0, w_idx)

        return ((s, t), d, w)

    def process_batch(self, data: TrainingSample, device: torch.device):
        """
        Flatten the batch of data into the format expected by the model. This
        will also move the data to the device.
        """

        inp, d, w = data
        s, t = inp

        s = torch.flatten(s, start_dim=0, end_dim=1)
        t = torch.flatten(t, start_dim=0, end_dim=1)
        d = torch.flatten(d, start_dim=0, end_dim=1)
        w = torch.flatten(w, start_dim=0, end_dim=1)

        s = s.to(device)
        t = t.to(device)
        d = d.to(device)
        w = w.to(device)

        return s, t, d, w
