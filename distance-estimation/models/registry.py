"""
The registry of all the models we can train.

This has to go in a separate file due to circular imports.
"""

from typing import Type

from models import IModelMetadata
from models.regression.linear_concat import RegLinearConcatModelMetadata
from models.regression.fcl_hadamard import RegFClHadamardModelMetadata
from models.regression.fcl_concat import RegFClConcatModelMetadata
from models.regression.fclsp_concat import RegFClSpConcatModelMetadata
from models.regression.resfcl_concat import RegResFClConcatModelMetadata
from models.regression.resmfcl_concat import RegResmFClConcatModelMetadata
from models.categorical.linear_concat import CatLinearConcatModelMetadata
from models.categorical.fcl_concat import CatFClConcatModelMetadata

MODEL_REGISTRY: dict[str, Type[IModelMetadata]] = {
    "reg-linear-concat": RegLinearConcatModelMetadata,
    "reg-fcl-hadamard": RegFClHadamardModelMetadata,
    "reg-fcl-concat": RegFClConcatModelMetadata,
    "reg-fclsp-concat": RegFClSpConcatModelMetadata,
    "reg-resfcl-concat": RegResFClConcatModelMetadata,
    "reg-resmfcl-concat": RegResmFClConcatModelMetadata,
    "cat-linear-concat": CatLinearConcatModelMetadata,
    "cat-fcl-concat": CatFClConcatModelMetadata,
}
