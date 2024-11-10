"""Defines the schema all distance estimation models should follow."""

import abc
import argparse

import torch


class IModel(abc.ABC, torch.nn.Module):
    """
    Base class for distance estimation models. They need to have some common
    fields for evaluation.
    """

    embedding_length: int
    max_distance: int

    def __init__(self, embedding_length: int, max_distance: int):
        super().__init__()
        self.embedding_length = embedding_length
        self.max_distance = max_distance


class IModelMetadata(abc.ABC):
    """
    Metadata for a distance estimation model. This includes the model itself,
    the loss function, and a way to extract the actual predictions from the
    model output.
    """

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        """
        Adds model-specific arguments to the argument parser.

        This method will be called after the script-specific arguments have been
        added.
        """
        pass

    @abc.abstractmethod
    def __init__(self, args: argparse.Namespace, class_weights: torch.Tensor):
        """
        Initializes the model metadata from command-line hyperparameters.

        We need the maximum distance for both types of models. Note that it's
        exclusive. We also provide the class weights becuase we need them for
        categorical models.
        """
        pass

    @abc.abstractmethod
    def get_model(self) -> IModel:
        """
        Returns the model.

        The model should take two parameters: the source and the target node
        embeddings. The output can be whatever - we'll use this class to turn it
        into a prediction.
        """
        pass

    @abc.abstractmethod
    def get_loss(self) -> torch.nn.Module:
        """
        Get the loss function to use.

        The loss function should take the model output, labels, and sample
        weights as input. All arguments are tensors whose first dimension is
        the batch size. This isn't a static method because the loss function may
        depend on the command-line arguments.
        """
        pass

    @abc.abstractmethod
    def extract_predictions(self, output: torch.Tensor) -> torch.Tensor:
        """
        Extracts the predictions from the model output.

        The argument is the batched output of the model, and the returned value
        should be a list of class labels for each batch, with shape (b, 1).
        """
        pass
