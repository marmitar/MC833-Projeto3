"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING, Union
from ...common import _aliases
from numpy import bool_ as bool
from ...common._typing import Array, Device, Dtype, NestedSequence, SupportsBufferProtocol

if TYPE_CHECKING:
    ...
isdtype = ...
unstack = ...
def astype(x: Array, dtype: Dtype, /, *, copy: bool = ..., device: Optional[Device] = ...) -> Array:
    """
    Array API compatibility wrapper for astype().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    ...

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: Optional[Dtype] = ..., device: Optional[Device] = ..., **kwargs) -> Array:
    """
    Array API compatibility wrapper for arange().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    ...

eye = ...
linspace = ...
UniqueAllResult = ...
UniqueCountsResult = ...
UniqueInverseResult = ...
unique_all = ...
unique_counts = ...
unique_inverse = ...
unique_values = ...
permute_dims = ...
std = ...
var = ...
cumulative_sum = ...
cumulative_prod = ...
empty = ...
empty_like = ...
full = ...
full_like = ...
ones = ...
ones_like = ...
zeros = ...
zeros_like = ...
reshape = ...
matrix_transpose = ...
vecdot = ...
nonzero = ...
ceil = ...
floor = ...
trunc = ...
matmul = ...
tensordot = ...
sign = ...
def asarray(obj: Union[Array, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol,], /, *, dtype: Optional[Dtype] = ..., device: Optional[Device] = ..., copy: Optional[Union[bool, np._CopyMode]] = ..., **kwargs) -> Array:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    ...

def clip(x: Array, /, min: Optional[Union[int, float, Array]] = ..., max: Optional[Union[int, float, Array]] = ...) -> Array:
    """
    Array API compatibility wrapper for clip().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    ...

def sort(x: Array, /, *, axis: int = ..., descending: bool = ..., stable: bool = ...) -> Array:
    """
    Array API compatibility layer around the lack of sort() in Dask.

    Warnings
    --------
    This function temporarily rechunks the array along `axis` to a single chunk.
    This can be extremely inefficient and can lead to out-of-memory errors.

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    ...

def argsort(x: Array, /, *, axis: int = ..., descending: bool = ..., stable: bool = ...) -> Array:
    """
    Array API compatibility layer around the lack of argsort() in Dask.

    See the corresponding documentation in the array library and/or the array API
    specification for more details.

    Warnings
    --------
    This function temporarily rechunks the array along `axis` into a single chunk.
    This can be extremely inefficient and can lead to out-of-memory errors.
    """
    ...

def count_nonzero(x: Array, axis=..., keepdims=...) -> Array:
    ...

__all__ = _aliases.__all__ + ['__array_namespace_info__', 'asarray', 'astype', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_right_shift', 'concat', 'pow', 'iinfo', 'finfo', 'can_cast', 'result_type', 'bool', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'complex64', 'complex128', 'iinfo', 'finfo', 'can_cast', 'count_nonzero', 'result_type']
_all_ignore = ...
