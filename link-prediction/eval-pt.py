"""
Evaluate a PyTorch link prediction model.

This is just a sanity test to make sure the conversion code worked as expected.
"""

import argparse
from typing import Any

import tensorflow as tf
import torch
import yaml

import util


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:

    model = torch.jit.load(args.model)
    model.eval()

    train, val, test = util.get_datasets(
        config["training-data"]["link-prediction"],
        args.num_train_samples,
        util.example_decoder(model.embedding_length),
    )

    def run(dataset: tf.data.Dataset) -> float:
        correct = 0
        total = 0

        for x, y in dataset.batch(util.BATCH_SIZE):
            s, t = x
            s = torch.Tensor(s.numpy())
            t = torch.Tensor(t.numpy())
            y = torch.Tensor(y.numpy())

            y_pred = model(s, t)
            correct += (y_pred > 0.5).eq(y > 0.5).sum().item()
            total += y.shape[0]

        return correct / total

    train_acc = run(train)
    val_acc = run(val)
    test_acc = run(test)

    print(f"Training set accuracy:   {train_acc:.4f}")
    print(f"Validation set accuracy: {val_acc:.4f}")
    print(f"Test set accuracy:       {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a link prediction model")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-t",
        "--num-train-samples",
        type=int,
        help="Number of samples used when training the model",
        default=10000,
    )
    parser.add_argument("model", type=str, help="Model to evaluate")

    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    main(args, config)
