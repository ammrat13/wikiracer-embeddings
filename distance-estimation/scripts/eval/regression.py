"""Evaluate a categorgical distance estimation model."""

import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import yaml

# See: https://stackoverflow.com/a/49155631
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data import DistanceEstimationDataset


def main(args: argparse.Namespace, config: dict[str, Any]):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.jit.load(args.model)
    model.eval()
    model.to(device)

    dataset_base = config["training-data"]["distance-estimation"]
    dataset = DistanceEstimationDataset(
        os.path.join(dataset_base, args.dataset),
        config["data"]["embeddings"][1536],
        max_distance=model.max_distance,
        num_runs=args.runs,
        embedding_length=model.embedding_length,
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, num_workers=2, pin_memory=True
    )

    print("Starting evaluation!\n")
    count_connected = 0
    count_unconnected = 0
    mean_absolute_error = 0
    mean_relative_error = 0
    mean_unconnected_prediction = 0

    with torch.no_grad():
        for data in tqdm(dataset_loader):
            s, t, labels, _ = dataset.process_batch(data, device)
            out = model(s, t)

            connected_mask = labels != 0
            count_connected += torch.sum(connected_mask).item()
            count_unconnected += len(labels) - count_connected

            mean_absolute_error += torch.sum(
                torch.where(
                    connected_mask,
                    torch.abs(out - labels),
                    0.0,
                )
            ).item()
            mean_relative_error += torch.sum(
                torch.where(
                    connected_mask,
                    torch.abs(out - labels) / torch.where(connected_mask, labels, 1.0),
                    0.0,
                )
            ).item()
            mean_unconnected_prediction += torch.sum(
                torch.where(
                    connected_mask,
                    0.0,
                    out,
                )
            ).item()
    mean_absolute_error /= count_connected
    mean_relative_error /= count_connected
    mean_unconnected_prediction /= count_unconnected

    print(f"    Mean Absolute Error:         {mean_absolute_error}")
    print(f"    Mean Relative Error:         {mean_relative_error}")
    print(f"    Mean Unconnected Prediction: {mean_unconnected_prediction}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a categorical model.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        help="How many BFS runs to use",
        default=None,
    )
    parser.add_argument(
        "model",
        type=argparse.FileType("rb"),
        help="Path to the model TorchScript file",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset to use",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    plt.style.use(config["plotting"]["style"])

    main(args, config)
