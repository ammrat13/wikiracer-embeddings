"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.categorical.logistic_mn_hadamard import LogisticMnHadamardModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "logistic_mn-hadamard": LogisticMnHadamardModelMetadata,
}
