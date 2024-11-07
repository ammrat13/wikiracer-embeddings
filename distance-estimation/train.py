"""Train a distance estimation model from the registry."""

import argparse
import sys
import os
from typing import Any, Type

import torch
from tqdm import tqdm
import yaml

from data import DistanceEstimationDataset
from models import IModelMetadata
from models.registry import MODEL_REGISTRY


def main(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_meta_cls: Type[IModelMetadata],
) -> None:

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Note that we can't shuffle the datasets, nor can we do the traditional
    # splitting. Just holding the random indicies in memory is too much. So, we
    # have separate files for each set.
    dataset_base = config["training-data"]["distance-estimation"]
    train_dataset = DistanceEstimationDataset(
        os.path.join(dataset_base, args.train_name),
        config["data"]["embeddings"][1536],
        max_distance=args.max_dist,
        num_runs=args.training_runs,
        embedding_length=args.embedding_length,
    )
    val_dataset = DistanceEstimationDataset(
        os.path.join(dataset_base, args.validation_name),
        config["data"]["embeddings"][1536],
        max_distance=args.max_dist,
        num_runs=args.validation_runs,
        embedding_length=args.embedding_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=2, pin_memory=True
    )

    model_meta = model_meta_cls(args, train_dataset.class_weights)
    model = model_meta.get_model().to(device)
    loss = model_meta.get_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    print("Starting training loop!\n")
    start_epoch = 0
    if args.continue_training:
        checkpoint = torch.load(
            args.checkpoint_output,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Continuing from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1} / {args.epochs}")

        train_loss = 0.0
        train_correct = 0
        train_count = 0
        model.train()
        for data in tqdm(train_loader):
            s, t, d, w = train_dataset.process_batch(data, device)
            optimizer.zero_grad()

            out = model(s, t)
            l = loss(out, d, w)
            l.backward()
            optimizer.step()

            p = model_meta.extract_predictions(out)
            train_loss += l.item()
            train_correct += torch.sum(p == d).item()
            train_count += len(p)
        train_accuracy = train_correct / train_count
        train_loss /= len(train_loader)

        val_correct = 0
        val_count = 0
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                s, t, d, w = val_dataset.process_batch(data, device)

                out = model(s, t)
                l = loss(out, d, w)
                val_loss += l.item()

                p = model_meta.extract_predictions(out)

                val_correct += torch.sum(p == d).item()
                val_count += len(p)

                val_connected_false_positives += torch.sum((p != 0) & (d == 0)).item()
                val_connected_false_negatives += torch.sum((p == 0) & (d != 0)).item()

                mtch = (p != 0) & (d != 0)
                val_connected_match += torch.sum(mtch).item()
                val_absolute_error += torch.sum(
                    torch.abs(torch.where(mtch, p - d, 0))
                ).item()
                val_relative_error += torch.sum(
                    torch.abs(torch.where(mtch, p - d, 0) / torch.where(d != 0, d, 1))
                ).item()
        val_accuracy = val_correct / val_count
        val_loss /= len(val_loader)

        print(f"    Training loss:           {train_loss}")
        print(f"    Training accuracy:       {train_accuracy}")
        print(f"    Validation loss:         {val_loss}")
        print(f"    Validation accuracy:     {val_accuracy}")
        print()

        # See: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            args.checkpoint_output,
        )
        # Also save the script for the model.
        torch.jit.script(model).save(args.script_output)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Specify the model name.", file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]
    if model_name not in MODEL_REGISTRY:
        print(f"Model {model_name} not found.", file=sys.stderr)
        sys.exit(1)
    model_meta_cls = MODEL_REGISTRY[model_name]
    sys.argv = sys.argv[0:1] + sys.argv[2:]

    parser = argparse.ArgumentParser(description="Train a distance estimation model.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-m",
        "--max-dist",
        type=int,
        help="Maximum distance to predict (exclusive)",
        default=7,
    )
    parser.add_argument(
        "-l",
        "--embedding-length",
        type=int,
        help="Number of features to use",
        default=512,
    )
    parser.add_argument(
        "--training-runs",
        type=int,
        help="How many BFS runs to use during training",
        default=100,
    )
    parser.add_argument(
        "--validation-runs",
        type=int,
        help="How many BFS runs to use during validation",
        default=10,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of epochs to train",
        default=100,
    )
    parser.add_argument(
        "-K",
        "--checkpoint-output",
        type=str,
        help="Where to save checkpoints",
        default="checkpoint.pt",
    )
    parser.add_argument(
        "-o",
        "--script-output",
        type=str,
        help="Output model in TorchScript",
        default="model.pt",
    )
    parser.add_argument(
        "-k",
        "--continue-training",
        action="store_true",
        help="Continue from a checkpoint",
    )
    parser.add_argument("train_name", type=str, help="Name of the training file")
    parser.add_argument("validation_name", type=str, help="Name of the validation file")
    model_meta_cls.add_args(parser)

    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    main(args, config, model_meta_cls)
