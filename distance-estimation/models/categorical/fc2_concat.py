import argparse
from typing import Any

import torch

from models import IModel
from models.categorical import ICategoricalModelMetadata


class CatFC2ConcatModelMetadata(ICategoricalModelMetadata):
    """
    Model with a 2-layer fully-connected network with concatenation to combine
    features.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        ICategoricalModelMetadata.add_args(parser)
        parser.add_argument(
            "--hidden-length",
            type=int,
            help="Length of the hidden layer",
            default=32,
        )

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)
        self.embedding_length = args.embedding_length
        self.hidden_length = args.hidden_length

    def get_model(self) -> IModel:
        return CatFC2ConcatModel(
            self.embedding_length, self.max_distance, self.hidden_length
        )

    def get_wandb_config(self) -> dict[str, Any]:
        return {"hidden_length": self.hidden_length}


@torch.compile
class CatFC2ConcatModel(IModel):

    def __init__(
        self,
        embedding_length: int,
        max_distance: int,
        hidden_length: int,
    ):
        super().__init__(embedding_length, max_distance)
        self.hidden_linear = torch.nn.Linear(2 * embedding_length, hidden_length)
        self.hidden_values = torch.nn.ReLU()
        self.output_linear = torch.nn.Linear(hidden_length, max_distance)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat((s, t), dim=1)
        x = self.hidden_linear(x)
        x = self.hidden_values(x)
        x = self.output_linear(x)
        return x
