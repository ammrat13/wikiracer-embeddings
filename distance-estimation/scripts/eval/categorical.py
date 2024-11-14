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
    correct = 0
    count = 0
    connected_false_positives = 0
    connected_false_negatives = 0
    connected_match = 0
    confusion_matrix = np.zeros((model.max_distance, model.max_distance))

    with torch.no_grad():
        for data in tqdm(dataset_loader):
            s, t, labels, _ = dataset.process_batch(data)
            out = model(s, t)
            preds = torch.argmax(out, dim=1)

            correct += torch.sum(preds == labels).item()
            count += len(labels)

            connected_false_positives += torch.sum((preds != 0) & (labels == 0)).item()
            connected_false_negatives += torch.sum((preds == 0) & (labels != 0)).item()
            connected_match += torch.sum((preds != 0) & (labels != 0)).item()

            confusion_matrix += sklearn.metrics.confusion_matrix(
                labels.detach().cpu(),
                preds.detach().cpu(),
                labels=list(range(model.max_distance)),
            )
    accuracy = correct / count
    connected_false_positives /= count
    connected_false_negatives /= count
    connected_match /= count
    confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)

    print(f"    Accuracy:     {accuracy}")
    print(f"    Connected FP: {connected_false_positives}")
    print(f"    Connected FN: {connected_false_negatives}")
    print(f"    Connected EQ: {connected_match}")
    print(f"    Confusion Matrix:")
    print(confusion_matrix)

    disp = sklearn.metrics.ConfusionMatrixDisplay(100 * confusion_matrix)
    disp.plot(include_values=True, values_format=".0f")
    plt.savefig(args.output)


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
        "--output",
        type=str,
        help="Where to write the confusion matrix",
        default="confusion_matrix.png",
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
