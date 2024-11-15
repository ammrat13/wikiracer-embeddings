"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.categorical.fc1_hadamard import CatFC1HadamardModelMetadata
from models.categorical.fc2_hadamard import CatFC2HadamardModelMetadata
from models.categorical.fc2_concat import CatFC2ConcatModelMetadata
from models.categorical.fc3_concat import CatFC3ConcatModelMetadata
from models.regression.fc2_hadamard import RegFC2HadamardModelMetadata
from models.regression.fc3_concat import RegFC3ConcatModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "cat-fc1-hadamard": CatFC1HadamardModelMetadata,
    "cat-fc2-hadamard": CatFC2HadamardModelMetadata,
    "cat-fc2-concat": CatFC2ConcatModelMetadata,
    "cat-fc3-concat": CatFC3ConcatModelMetadata,
    "reg-fc2-hadamard": RegFC2HadamardModelMetadata,
    "reg-fc3-concat": RegFC3ConcatModelMetadata,
}
