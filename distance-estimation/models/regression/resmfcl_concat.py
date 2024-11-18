import argparse
from typing import Any

import torch

from models import IModel
from models.util import ResidualLayer
from models.regression import IRegressionModelMetadata


class RegResmFClConcatModelMetadata(IRegressionModelMetadata):
    """
    An l-layer fully-connected neural network with element-wise concatenation to
    combine features. Each layer has the same width, and there is a residual
    connection between the first and last hidden layers.
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
        return RegResmFClConcatModel(
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
class RegResmFClConcatModel(IModel):

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

        self.input_linear = torch.nn.Linear(2 * embedding_length, hidden_length)
        self.input_values = torch.nn.ReLU()

        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *(
                        [torch.nn.Linear(hidden_length, hidden_length)]
                        + ([torch.nn.ReLU()] if it != num_hidden_layers - 2 else [])
                    )
                )
                for it in range(num_hidden_layers - 1)
            ]
        )
        self.hidden_values = torch.nn.ReLU()

        self.output_linear = torch.nn.Linear(hidden_length, 1)
        self.output_values = torch.nn.Flatten(0)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat((s, t), dim=1)
        x = self.input_linear(x)
        x = self.input_values(x)

        x_res = x
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = x + x_res
        x = self.hidden_values(x)

        x = self.output_linear(x)
        x = self.output_values(x)
        return x
