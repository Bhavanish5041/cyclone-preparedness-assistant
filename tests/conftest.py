"""Global pytest fixtures and compatibility configurations for the test suite."""

import sys

# Evidently AI 0.7.23 references pandas.core.dtypes.common.infer_dtype_from_object,
# which was relocated/deprecated in pandas 3.x. Shim it globally for tests if missing.
try:
    import pandas.core.dtypes.common
    import pandas._libs.lib

    if not hasattr(pandas.core.dtypes.common, "infer_dtype_from_object"):
        pandas.core.dtypes.common.infer_dtype_from_object = pandas._libs.lib.infer_dtype
except Exception:
    pass
