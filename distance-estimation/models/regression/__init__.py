import abc
import argparse

import torch

from models import IModelMetadata


class IRegressionModelMetadata(IModelMetadata):
    """
    Distance estimation models that use MSE for their final layer.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        self.max_distance = args.max_dist

    def get_loss(self) -> torch.nn.Module:
        return RegressionModelLoss(self.max_distance)

    def extract_predictions(self, output: torch.Tensor) -> torch.Tensor:
        return output


@torch.compile
class RegressionModelLoss(torch.nn.Module):
    max_distance: int
    cutoff: float

    def __init__(self, max_distance: int):
        super().__init__()
        self.max_distance = max_distance
        self.cutoff = float(max_distance - 1)

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean(
            sample_weights
            * torch.where(
                labels == 0,
                torch.where(
                    output >= self.cutoff,
                    0.0,
                    (output - self.cutoff) ** 2,
                ),
                (output - labels) ** 2,
            )
        )
