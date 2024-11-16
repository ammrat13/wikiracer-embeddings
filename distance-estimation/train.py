"""Train a distance estimation model from the registry."""

import argparse
import math
import os
import sys
from typing import Any, Type

import torch
from tqdm import tqdm
import wandb
import yaml

from data import DistanceEstimationDataset
from models import IModelMetadata, IModel
from models.registry import MODEL_REGISTRY


def torch_setup() -> torch.device:
    """Do one-time setup for PyTorch."""
    # Silence warning about performance
    torch.set_float32_matmul_precision("high")
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def wandb_setup(
    args: argparse.Namespace, model_meta: IModelMetadata
) -> wandb.sdk.wandb_run.Run:
    """Do one-time setup for Weights & Biases."""

    if args.continue_training:
        checkpoint = torch.load(args.checkpoint_output, weights_only=True)
        run_id = checkpoint["run_id"]
        wandb.init(
            project="cs229-distance-estimation",
            group=args.model_name,
            id=run_id,
            resume="must",
        )

    return wandb.init(
        project="cs229-distance-estimation",
        group=args.model_name,
        config={
            "training_runs": {
                "bfs": args.training_bfs,
                "edge": args.training_edge,
            },
            "validation_runs": {
                "bfs": args.validation_bfs,
                "edge": args.validation_edge,
            },
            "max_dist": args.max_dist,
            "embedding_length": args.embedding_length,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "model_name": args.model_name,
            "model_config": model_meta.get_wandb_config(),
        },
    )


def get_data(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    DistanceEstimationDataset,
    DistanceEstimationDataset,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Get datasets and loaders for training and validation data.

    The loaders do not shuffle the data. It's in batches of BFS iterations
    anyway, and the nodes were selected randomly.
    """

    dataset_path = config["training-data"]["distance-estimation"]
    embeddings_path = config["data"]["embeddings"][1536]

    DATASET_ARGS = {
        "embedding_filename": embeddings_path,
        "embedding_length": args.embedding_length,
        "max_distance": args.max_dist,
        "device": device,
    }
    LOADER_ARGS = {
        "num_workers": 2,
        "pin_memory": True,
    }

    train_dataset = DistanceEstimationDataset(
        os.path.join(dataset_path, args.train_name),
        num_bfs=args.training_bfs,
        num_edge=args.training_edge,
        **DATASET_ARGS,
    )
    val_dataset = DistanceEstimationDataset(
        os.path.join(dataset_path, args.validation_name),
        num_bfs=args.validation_bfs,
        num_edge=args.validation_edge,
        **DATASET_ARGS,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **LOADER_ARGS)
    val_loader = torch.utils.data.DataLoader(val_dataset, **LOADER_ARGS)

    return train_dataset, val_dataset, train_loader, val_loader


def get_model(
    args: argparse.Namespace,
    device: torch.device,
    model_meta_cls: Type[IModelMetadata],
    class_weights: torch.Tensor,
) -> tuple[
    int,
    IModelMetadata,
    IModel,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """
    Create the model, and all of the associated training objects. Load from
    checkpoint if needed.
    """

    model_meta = model_meta_cls(args, class_weights)
    model = model_meta.get_model().to(device)
    loss = model_meta.get_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", threshold=args.reduction_threshold, patience=5
    )

    start_epoch = 0
    if args.continue_training:
        checkpoint = torch.load(args.checkpoint_output, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    return start_epoch, model_meta, model, loss, optimizer, scheduler


def main(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_meta_cls: Type[IModelMetadata],
) -> None:

    device = torch_setup()
    print(f"Using device: {device}")

    train_dset, val_dset, train_ldr, val_ldr = get_data(args, config, device)
    start_epoch, model_meta, model, loss, optimizer, scheduler = get_model(
        args,
        device,
        model_meta_cls,
        train_dset.class_weights,
    )

    run = wandb_setup(args, model_meta)

    print("Starting training loop!\n")
    if start_epoch != 0:
        print(f"Continuing from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1} / {args.epochs}")

        train_loss = 0.0
        model.train()
        for data in tqdm(train_ldr):
            s, t, d, w = train_dset.process_batch(data)
            optimizer.zero_grad()
            out = model(s, t)
            l = loss(out, d, w)
            l.backward()
            optimizer.step()
            train_loss += l.item()
        train_loss /= len(train_ldr)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_ldr):
                s, t, d, w = val_dset.process_batch(data)
                out = model(s, t)
                l = loss(out, d, w)
                val_loss += l.item()
        val_loss /= len(val_ldr)

        print(f"    Training loss:           {train_loss}")
        print(f"    Validation loss:         {val_loss}")
        print(f"    Learning rate:           {scheduler.get_last_lr()}")
        print()
        run.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # See: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        torch.save(
            {
                "epoch": epoch,
                "run_id": run.id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            args.checkpoint_output,
        )
        # Also save the script for the model.
        torch.jit.script(model).save(args.script_output)

        # Save to W&B
        art = wandb.Artifact(
            name=args.model_name,
            type="model",
            metadata={
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )
        art.add_file(args.checkpoint_output, name="checkpoint.pt")
        art.add_file(args.script_output, name="model.scr.pt")
        run.log_artifact(art)

        # Early stopping when the learning rate gets too low
        if (
            math.log10(args.learning_rate) - math.log10(scheduler.get_last_lr()[0])
            >= args.reduction_limit - 0.5
        ):
            print("Learning rate too low. Stopping training.")
            break
        scheduler.step(val_loss)

    wandb.finish()


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
        default=256,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=0.001,
    )
    parser.add_argument(
        "--reduction-threshold",
        type=float,
        help="Threshold for reducing learning rate",
        default=1e-2,
    )
    parser.add_argument(
        "--reduction-limit",
        type=int,
        help="Stop training when learning rate is reduced by this factor (log scale)",
        default=2,
    )
    parser.add_argument(
        "--training-bfs",
        type=int,
        help="How many BFS runs to use during training",
        default=None,
    )
    parser.add_argument(
        "--training-edge",
        type=int,
        help="How many additional edges to use during training",
        default=None,
    )
    parser.add_argument(
        "--validation-bfs",
        type=int,
        help="How many BFS runs to use during validation",
        default=None,
    )
    parser.add_argument(
        "--validation-edge",
        type=int,
        help="How many additional edges to use during validation",
        default=None,
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
        default="model.scr.pt",
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

    # This wasn't part of the argparse command-line arguments, so we have to add
    # it manually.
    args.model_name = model_name

    main(args, config, model_meta_cls)
