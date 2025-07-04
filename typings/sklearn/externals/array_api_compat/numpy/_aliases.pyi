"""
This type stub file was generated by pyright.
"""

import numpy as np
from ..common import _aliases
from typing import Optional, TYPE_CHECKING, Union
from ._typing import Device, Dtype, NestedSequence, SupportsBufferProtocol, ndarray

if TYPE_CHECKING:
    ...
bool = np.bool_
acos = ...
acosh = ...
asin = ...
asinh = ...
atan = ...
atan2 = ...
atanh = ...
bitwise_left_shift = ...
bitwise_invert = ...
bitwise_right_shift = ...
concat = ...
pow = ...
arange = ...
empty = ...
empty_like = ...
eye = ...
full = ...
full_like = ...
linspace = ...
ones = ...
ones_like = ...
zeros = ...
zeros_like = ...
UniqueAllResult = ...
UniqueCountsResult = ...
UniqueInverseResult = ...
unique_all = ...
unique_counts = ...
unique_inverse = ...
unique_values = ...
std = ...
var = ...
cumulative_sum = ...
cumulative_prod = ...
clip = ...
permute_dims = ...
reshape = ...
argsort = ...
sort = ...
nonzero = ...
ceil = ...
floor = ...
trunc = ...
matmul = ...
matrix_transpose = ...
tensordot = ...
sign = ...
def asarray(obj: Union[ndarray, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol,], /, *, dtype: Optional[Dtype] = ..., device: Optional[Device] = ..., copy: Optional[Union[bool, np._CopyMode]] = ..., **kwargs) -> ndarray:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    ...

def astype(x: ndarray, dtype: Dtype, /, *, copy: bool = ..., device: Optional[Device] = ...) -> ndarray:
    ...

def count_nonzero(x: ndarray, axis=..., keepdims=...) -> ndarray:
    ...

if hasattr(np, 'vecdot'):
    vecdot = ...
else:
    vecdot = ...
if hasattr(np, 'isdtype'):
    isdtype = ...
else:
    isdtype = ...
if hasattr(np, 'unstack'):
    unstack = ...
else:
    unstack = ...
__all__ = _aliases.__all__ + ['__array_namespace_info__', 'asarray', 'astype', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_right_shift', 'bool', 'concat', 'count_nonzero', 'pow']
_all_ignore = ...
