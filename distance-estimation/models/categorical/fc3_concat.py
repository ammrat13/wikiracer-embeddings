import argparse

import torch

from models import IModel
from models.categorical import ICategoricalModelMetadata


class CatFC3ConcatModelMetadata(ICategoricalModelMetadata):
    """
    Model with a 3-layer fully-connected network with concatenation to combine
    features.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        ICategoricalModelMetadata.add_args(parser)
        parser.add_argument(
            "--hidden-length-1",
            type=int,
            help="Length of the first hidden layer",
            default=64,
        )
        parser.add_argument(
            "--hidden-length-2",
            type=int,
            help="Length of the second hidden layer",
            default=32,
        )
        parser.add_argument(
            "--dropout-rate",
            type=float,
            help="Dropout rate for the model",
            default=0.0,
        )

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)
        self.embedding_length = args.embedding_length
        self.hidden_length_1 = args.hidden_length_1
        self.hidden_length_2 = args.hidden_length_2
        self.dropout_rate = args.dropout_rate

    def get_model(self) -> IModel:
        return CatFC3ConcatModel(
            self.embedding_length,
            self.max_distance,
            self.hidden_length_1,
            self.hidden_length_2,
            self.dropout_rate,
        )

    def get_wandb_config(self) -> dict[str, any]:
        return {
            "hidden_length_1": self.hidden_length_1,
            "hidden_length_2": self.hidden_length_2,
            "dropout_rate": self.dropout_rate,
        }


@torch.compile
class CatFC3ConcatModel(IModel):

    def __init__(
        self,
        embedding_length: int,
        max_distance: int,
        hidden_length_1: int,
        hidden_length_2: int,
        dropout_rate: float,
    ):
        super().__init__(embedding_length, max_distance)
        self.hidden_1_linear = torch.nn.Linear(2 * embedding_length, hidden_length_1)
        self.hidden_1_values = torch.nn.ReLU()
        self.hidden_1_dropout = torch.nn.Dropout(dropout_rate)
        self.hidden_2_linear = torch.nn.Linear(hidden_length_1, hidden_length_2)
        self.hidden_2_values = torch.nn.ReLU()
        self.hidden_2_dropout = torch.nn.Dropout(dropout_rate)
        self.output_linear = torch.nn.Linear(hidden_length_2, max_distance)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat((s, t), dim=1)
        x = self.hidden_1_linear(x)
        x = self.hidden_1_values(x)
        x = self.hidden_1_dropout(x)
        x = self.hidden_2_linear(x)
        x = self.hidden_2_values(x)
        x = self.hidden_2_dropout(x)
        x = self.output_linear(x)
        return x
