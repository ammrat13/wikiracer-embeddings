import argparse

import torch

from models.categorical import ICategoricalModelMetadata


class LogisticMnHadamardModelMetadata(ICategoricalModelMetadata):
    """
    Model that uses multinomial logistic regression with element-wise
    multiplication to combine features.
    """

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        super().__init__(args, class_weights)
        self.embedding_length = args.embedding_length

    def get_model(self) -> torch.nn.Module:
        return LogisticMnHadamardModel(self.embedding_length, self.max_distance)


@torch.compile
class LogisticMnHadamardModel(torch.nn.Module):
    embedding_length: int
    max_distance: int

    def __init__(self, embedding_length: int, max_distance: int):
        super().__init__()
        self.embedding_length = embedding_length
        self.max_distance = max_distance
        self.linear = torch.nn.Linear(embedding_length, max_distance)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = s * t
        x = self.linear(x)
        return x
