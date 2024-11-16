"""
Evaluate a categorgical distance estimation model by turning it into a
regression model by expected value.
"""

import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import wandb
import yaml

# See: https://stackoverflow.com/a/49155631
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data import DistanceEstimationDataset


def main(args: argparse.Namespace, config: dict[str, Any]):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.jit.load(args.model_path)
    model.eval()
    model.to(device)

    dataset_base = config["training-data"]["distance-estimation"]
    dataset = DistanceEstimationDataset(
        os.path.join(dataset_base, args.dataset),
        config["data"]["embeddings"][1536],
        embedding_length=model.embedding_length,
        max_distance=model.max_distance,
        num_bfs=args.bfs,
        num_edge=args.edges,
        device=device,
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

    hist_bins = np.linspace(0, model.max_distance, args.histogram_bins + 1)
    histogram = np.zeros((model.max_distance - 1, args.histogram_bins), dtype=np.int32)

    exval_coefs = torch.arange(model.max_distance, dtype=torch.float32, device=device)
    exval_coefs[0] = model.max_distance

    with torch.no_grad():
        for data in tqdm(dataset_loader):
            s, t, labels, _ = dataset.process_batch(data)
            out = model(s, t)
            pred = torch.softmax(out, dim=1) @ exval_coefs

            batch_connected_mask = labels != 0
            batch_connected = torch.sum(batch_connected_mask).item()
            batch_unconnected = len(labels) - batch_connected
            count_connected += batch_connected
            count_unconnected += batch_unconnected

            mean_absolute_error += torch.sum(
                torch.where(
                    batch_connected_mask,
                    torch.abs(pred - labels),
                    0.0,
                )
            ).item()
            mean_relative_error += torch.sum(
                torch.where(
                    batch_connected_mask,
                    torch.abs(pred - labels)
                    / torch.where(batch_connected_mask, labels, 1.0),
                    0.0,
                )
            ).item()
            mean_unconnected_prediction += torch.sum(
                torch.where(
                    batch_connected_mask,
                    0.0,
                    pred,
                )
            ).item()

            for it in range(model.max_distance):
                mask = labels == it
                batch_hist, _ = np.histogram(
                    pred[mask].cpu().numpy(),
                    bins=hist_bins,
                )
                histogram[it] += batch_hist

    mean_absolute_error /= count_connected
    mean_relative_error /= count_connected
    mean_unconnected_prediction /= count_unconnected

    print(f"    Mean Absolute Error:         {mean_absolute_error}")
    print(f"    Mean Relative Error:         {mean_relative_error}")
    print(f"    Mean Unconnected Prediction: {mean_unconnected_prediction}")

    histogram = histogram.astype(np.float32)
    histogram /= np.sum(histogram, axis=1, keepdims=True)
    histogram *= args.histogram_bins / model.max_distance
    for it in range(model.max_distance):
        fig, ax = plt.subplots()
        ax.bar(
            hist_bins[:-1],
            histogram[it],
            width=model.max_distance / args.histogram_bins,
            align="edge",
        )
        ax.set_title(f"Distance Histogram for \\(d = {it}\\)")
        ax.set_xlabel("Predicted Distance")
        ax.set_ylabel("Probability Density")
        fig.savefig(
            os.path.join(args.histogram_dir, f"histogram_{it}.png"),
            bbox_inches="tight",
        )


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
        "--bfs",
        type=int,
        help="How many BFS runs to use",
        default=None,
    )
    parser.add_argument(
        "--edges",
        type=int,
        help="How many additional edges to use",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--histogram-dir",
        type=str,
        help="Output directory for the histograms",
        default=".",
    )
    parser.add_argument(
        "-b",
        "--histogram-bins",
        type=int,
        help="Number of bins for the histograms",
        default=1000,
    )
    parser.add_argument(
        "artifact",
        type=str,
        help="Name of the W&B model artifact",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset to use",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    plt.style.use(config["plotting"]["style"])

    api = wandb.Api()
    entity = config["wandb"]["entity"]
    project = config["wandb"]["projects"]["distance-estimation"]
    artifact = api.artifact(f"{entity}/{project}/{args.artifact}")
    model_folder = artifact.download(path_prefix="model.scr.pt")
    args.model_path = os.path.join(model_folder, "model.scr.pt")

    main(args, config)