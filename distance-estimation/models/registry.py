"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.categorical.logistic_mn_hadamard import LogisticMnHadamardModelMetadata
from models.categorical.fc2_ce_hadamard import FC2CEHadamardModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "logistic_mn-hadamard": LogisticMnHadamardModelMetadata,
    "fc2_ce-hadamard": FC2CEHadamardModelMetadata,
}
