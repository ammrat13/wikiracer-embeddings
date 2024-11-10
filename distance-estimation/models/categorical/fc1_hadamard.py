import argparse

import torch

from models import IModel
from models.categorical import ICategoricalModelMetadata


class CatFC1HadamardModelMetadata(ICategoricalModelMetadata):
    """
    Model that uses multinomial logistic regression with element-wise
    multiplication to combine features.
    """

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)
        self.embedding_length = args.embedding_length

    def get_model(self) -> IModel:
        return CatFC1HadamardModel(self.embedding_length, self.max_distance)


@torch.compile
class CatFC1HadamardModel(IModel):

    def __init__(self, embedding_length: int, max_distance: int):
        super().__init__(embedding_length, max_distance)
        self.linear = torch.nn.Linear(embedding_length, max_distance)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = s * t
        x = self.linear(x)
        return x
