import argparse
from typing import Any

import torch

from models import IModel
from models.regression import IRegressionModelMetadata


class RegFClHadamardModelMetadata(IRegressionModelMetadata):
    """
    An l-layer fully-connected neural network with element-wise multiplication
    to combine features.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        IRegressionModelMetadata.add_args(parser)
        parser.add_argument(
            "--hidden-layers",
            type=int,
            help="Number of hidden layers",
            default=2,
        )
        parser.add_argument(
            "--hidden-length",
            type=int,
            help="Length of each hidden layer",
            default=256,
        )

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)
        self.num_hidden_layers = args.hidden_layers
        self.hidden_length = args.hidden_length

    def get_model(self) -> IModel:
        return RegFClHadamardModel(
            self.embedding_length,
            self.max_distance,
            self.num_hidden_layers,
            self.hidden_length,
        )

    def get_wandb_config(self) -> dict[str, Any]:
        return {
            "hidden_layers": self.num_hidden_layers,
            "hidden_length": self.hidden_length,
        }


@torch.compile
class RegFClHadamardModel(IModel):

    def __init__(
        self,
        embedding_length: int,
        max_distance: int,
        num_hidden_layers: int,
        hidden_length: int,
    ):
        super().__init__(embedding_length, max_distance)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_length = hidden_length

        self.input_linear = torch.nn.Linear(embedding_length, hidden_length)
        self.input_values = torch.nn.ReLU()

        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_length, hidden_length),
                    torch.nn.ReLU(),
                )
                for _ in range(num_hidden_layers - 1)
            ]
        )

        self.output_linear = torch.nn.Linear(hidden_length, 1)
        self.output_values = torch.nn.Flatten(0)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = s * t
        x = self.input_linear(x)
        x = self.input_values(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_linear(x)
        x = self.output_values(x)
        return x
