import argparse
from typing import Any

import torch

from models import IModel
from models.regression import IRegressionModelMetadata


class RegLinearHadamardModelMetadata(IRegressionModelMetadata):
    """
    Linear regression with element-wise multiplication to combine features.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        IRegressionModelMetadata.add_args(parser)

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)

    def get_model(self) -> IModel:
        return RegLinearHadamardModel(
            self.embedding_length,
            self.max_distance,
        )


@torch.compile
class RegLinearHadamardModel(IModel):

    def __init__(self, embedding_length: int, max_distance: int):
        super().__init__(embedding_length, max_distance)
        self.linear = torch.nn.Linear(embedding_length, 1)
        self.flatten = torch.nn.Flatten(0)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = s * t
        x = self.linear(x)
        x = self.flatten(x)
        return x
