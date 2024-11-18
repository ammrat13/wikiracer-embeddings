"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.regression.fcl_concat import RegFClConcatModelMetadata
from models.regression.resfcl_concat import RegResFClConcatModelMetadata
from models.regression.resmfcl_concat import RegResmFClConcatModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "reg-fcl-concat": RegFClConcatModelMetadata,
    "reg-resfcl-concat": RegResFClConcatModelMetadata,
    "reg-resmfcl-concat": RegResmFClConcatModelMetadata,
}
