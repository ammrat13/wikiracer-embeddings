"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.categorical.multinomial_hadamard import MultinomialHadamardModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "logistic_mn-hadamard": MultinomialHadamardModelMetadata,
}
