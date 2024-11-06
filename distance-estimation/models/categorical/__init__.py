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
        self.max_distance = args.max_dist
        self.class_weights = class_weights

    @abc.abstractmethod
    def get_model(self) -> torch.nn.Module:
        pass

    def get_loss(self) -> torch.nn.Module:
        return CategoricalModelLoss(self.max_distance, self.class_weights)

    def extract_predictions(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, dim=1)


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
        trunc = torch.where(
            labels >= self.max_distance,
            torch.tensor(0, dtype=torch.uint8),
            labels,
        ).long()
        return self.cross_entropy(output, trunc)
