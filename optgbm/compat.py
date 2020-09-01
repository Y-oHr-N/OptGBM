"""Compatibility library."""

import lightgbm as lgb
import sklearn

if lgb.__version__ >= "2.2.2" and lgb.__version__ < "3.0.0":
    from lightgbm.engine import _CVBooster  # noqa
else:
    from lightgbm.engine import CVBooster as _CVBooster  # noqa

if lgb.__version__ >= "2.3":
    from lightgbm.sklearn import _EvalFunctionWrapper  # noqa
    from lightgbm.sklearn import _ObjectiveFunctionWrapper  # noqa
else:
    from lightgbm.sklearn import (  # noqa
        _eval_function_wrapper as _EvalFunctionWrapper,
    )
    from lightgbm.sklearn import (  # noqa
        _objective_function_wrapper as _ObjectiveFunctionWrapper,
    )

if sklearn.__version__ >= "0.22":
    from sklearn.utils import _safe_indexing  # noqa
else:
    from sklearn.utils import safe_indexing as _safe_indexing  # noqa
