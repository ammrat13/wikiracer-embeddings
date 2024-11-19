"""
Evaluate how well the expected values from a categorical model are reflected
in the outputs of a regression model.
"""

import argparse
import math
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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

    reg_model = torch.jit.load(args.reg_model_path)
    reg_model.eval()
    reg_model.to(device)
    cat_model = torch.jit.load(args.cat_model_path)
    cat_model.eval()
    cat_model.to(device)

    assert reg_model.embedding_length == cat_model.embedding_length
    assert reg_model.max_distance == cat_model.max_distance

    dataset_base = config["training-data"]["distance-estimation"]
    dataset = DistanceEstimationDataset(
        os.path.join(dataset_base, args.dataset),
        config["data"]["embeddings"][1536],
        embedding_length=reg_model.embedding_length,
        max_distance=reg_model.max_distance,
        num_bfs=args.bfs,
        num_edge=args.edges,
        device=device,
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, num_workers=2, pin_memory=True
    )

    print("Starting evaluation!\n")
    count = 0
    deviation = 0.0

    exval_kernel = torch.arange(reg_model.max_distance, device=device).float()
    exval_kernel[0] = reg_model.max_distance

    with torch.no_grad():
        for data in tqdm(dataset_loader):
            s, t, labels, sample_weights = dataset.process_batch(data)

            reg_out = reg_model(s, t)
            reg_pred = reg_out + 1.0

            cat_out = cat_model(s, t)
            cat_out = torch.softmax(cat_out, dim=1)
            cat_pred = torch.sum(cat_out * exval_kernel, dim=1)

            count += len(labels)
            deviation += torch.sum(sample_weights * (reg_pred - cat_pred) ** 2).item()

    deviation /= count

    print(f"    Mean Squared Deviation: {deviation}")
    print(f"    Mean Deviation:         {math.sqrt(deviation)}")


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
        "regression_artifact",
        type=str,
        help="Name of the W&B regression model artifact",
    )
    parser.add_argument(
        "categorical_artifact",
        type=str,
        help="Name of the W&B categorical model artifact",
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
    reg_artifact = api.artifact(f"{entity}/{project}/{args.regression_artifact}")
    cat_artifact = api.artifact(f"{entity}/{project}/{args.categorical_artifact}")
    reg_model_folder = reg_artifact.download(path_prefix="model.scr.pt")
    cat_model_folder = cat_artifact.download(path_prefix="model.scr.pt")
    args.reg_model_path = os.path.join(reg_model_folder, "model.scr.pt")
    args.cat_model_path = os.path.join(cat_model_folder, "model.scr.pt")

    main(args, config)
