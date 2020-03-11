"""Type hints."""

from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from scipy.sparse import spmatrix
from sklearn.model_selection import BaseCrossValidator

if lgb.__version__ >= "2.2.2":
    from lightgbm.engine import _CVBooster
else:
    from lightgbm.engine import CVBooster as _CVBooster

CVType = Union[BaseCrossValidator, int, List[Tuple]]

LightGBMCallbackEnvType = NamedTuple(
    "LightGBMCallbackEnv",
    [
        ("model", _CVBooster),
        ("params", Dict[str, Any]),
        ("iteration", int),
        ("begin_iteration", int),
        ("end_iteration", int),
        ("evaluation_result_list", List),
    ],
)

OneDimArrayLikeType = Union[np.ndarray, pd.Series]
TwoDimArrayLikeType = Union[np.ndarray, pd.DataFrame, spmatrix]

RandomStateType = Union[int, np.random.RandomState]
