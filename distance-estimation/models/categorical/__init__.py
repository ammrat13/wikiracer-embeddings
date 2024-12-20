import abc
import argparse

import torch

from models import IModelMetadata


class ICategoricalModelMetadata(IModelMetadata):
    """
    Distance estimation models that use soft-max for their final layer.

    Note that the model actually doesn't do the soft-max for us. This is because
    the categorical cross entropy loss function expects the raw logits.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        self.embedding_length = args.embedding_length
        self.max_distance = args.max_dist
        self.class_weights = class_weights

    def get_loss(self) -> torch.nn.Module:
        return CategoricalModelLoss(self.max_distance, self.class_weights)


@torch.compile
class CategoricalModelLoss(torch.nn.Module):
    max_distance: int

    def __init__(self, max_distance: int, class_weights: torch.Tensor):
        super().__init__()
        self.max_distance = max_distance
        self.cross_entropy = torch.nn.CrossEntropyLoss(class_weights)

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.cross_entropy(output, labels)
