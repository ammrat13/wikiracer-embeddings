"""Evaluate a categorgical distance estimation model."""

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import yaml

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
    correct = 0
    count = 0
    connected_false_positives = 0
    connected_false_negatives = 0
    connected_match = 0
    absolute_error = 0.0
    relative_error = 0.0
    confusion_matrix = np.zeros((model.max_distance, model.max_distance))

    with torch.no_grad():
        for data in tqdm(dataset_loader):
            s, t, labels, w = dataset.process_batch(data, device)
            out = model(s, t)
            preds = torch.argmax(out, dim=1)

            correct += torch.sum(preds == labels).item()
            count += len(labels)

            connected_false_positives += torch.sum((preds != 0) & (labels == 0)).item()
            connected_false_negatives += torch.sum((preds == 0) & (labels != 0)).item()

            conn_match = (preds != 0) & (labels != 0)
            connected_match += torch.sum(conn_match).item()
            absolute_error += torch.sum(
                torch.abs(torch.where(conn_match, preds - labels, 0))
            ).item()
            relative_error += torch.sum(
                torch.abs(
                    torch.where(conn_match, preds - labels, 0)
                    / torch.where(labels != 0, labels, 1)
                )
            ).item()

            if args.confusion_matrix:
                confusion_matrix += sklearn.metrics.confusion_matrix(
                    labels.detach().cpu(),
                    preds.detach().cpu(),
                    labels=list(range(model.max_distance)),
                )

    print(f"    Accuracy:     {correct / count}")
    print(f"    Connected FP: {connected_false_positives / count}")
    print(f"    Connected FN: {connected_false_negatives / count}")
    print(f"    Connected EQ: {connected_match / count}")
    print(f"    MAE:          {absolute_error / connected_match}")
    print(f"    MRE:          {relative_error / connected_match}")

    if args.confusion_matrix:
        confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)
        print(f"    Confusion Matrix:")
        print(confusion_matrix)

        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix)
        disp.plot(
            include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
        )
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
        "-r",
        "--runs",
        type=int,
        help="How many BFS runs to use",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--confusion-matrix",
        action="store_true",
        help="Whether to create a confusion matrix",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Where to write the confusion matrix",
        default="confusion_matrix.png",
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

    main(args, config)
