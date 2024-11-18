import argparse
from typing import Any

import torch

from models import IModel
from models.regression import IRegressionModelMetadata


class RegLinearConcatModelMetadata(IRegressionModelMetadata):
    """
    Linear regression with element-wise concatenation to combine features.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        IRegressionModelMetadata.add_args(parser)

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)

    def get_model(self) -> IModel:
        return RegLinearConcatModel(
            self.embedding_length,
            self.max_distance,
        )


@torch.compile
class RegLinearConcatModel(IModel):

    def __init__(self, embedding_length: int, max_distance: int):
        super().__init__(embedding_length, max_distance)
        self.linear = torch.nn.Linear(2 * embedding_length, 1)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat((s, t), dim=1)
        x = self.linear(x)
        return x
