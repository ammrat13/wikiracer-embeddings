"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.categorical.fc1_hadamard import CatFC1HadamardModelMetadata
from models.categorical.fc2_hadamard import CatFC2HadamardModelMetadata
from models.categorical.fc2_concat import CatFC2ConcatModelMetadata
from models.regression.fc2_hadamard import RegFC2HadamardModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "cat-fc1-hadamard": CatFC1HadamardModelMetadata,
    "cat-fc2-hadamard": CatFC2HadamardModelMetadata,
    "cat-fc2-concat": CatFC2ConcatModelMetadata,
    "reg-fc2-hadamard": RegFC2HadamardModelMetadata,
}
