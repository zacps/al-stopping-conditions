import numpy as np

import matlab
# This is actually `matlab._internal`, but matlab/__init__.py
# mangles the path making it appear as `_internal`.
# Importing it under a different name would be a bad idea.
from _internal.mlarray_utils import _get_strides, _get_mlsize

def _wrapper__init__(self, arr):
    assert arr.dtype == type(self)._numpy_type
    self._python_type = type(arr.dtype.type().item())
    self._is_complex = np.issubdtype(arr.dtype, np.complexfloating)
    self._size = _get_mlsize(arr.shape)
    self._strides = _get_strides(self._size)[:-1]
    self._start = 0

    if self._is_complex:
        self._real = arr.real.ravel(order='F')
        self._imag = arr.imag.ravel(order='F')
    else:
        self._data = arr.ravel(order='F')

_wrappers = {}
def _define_wrapper(matlab_type, numpy_type):
    t = type(matlab_type.__name__, (matlab_type,), dict(
        __init__=_wrapper__init__,
        _numpy_type=numpy_type
    ))
    # this tricks matlab into accepting our new type
    t.__module__ = matlab_type.__module__
    _wrappers[numpy_type] = t

_define_wrapper(matlab.double, np.double)
_define_wrapper(matlab.single, np.single)
_define_wrapper(matlab.uint8, np.uint8)
_define_wrapper(matlab.int8, np.int8)
_define_wrapper(matlab.uint16, np.uint16)
_define_wrapper(matlab.int16, np.int16)
_define_wrapper(matlab.uint32, np.uint32)
_define_wrapper(matlab.int32, np.int32)
_define_wrapper(matlab.uint64, np.uint64)
_define_wrapper(matlab.int64, np.int64)
_define_wrapper(matlab.logical, np.bool_)

def as_matlab(arr):
    try:
        cls = _wrappers[arr.dtype.type]
    except KeyError:
        raise TypeError("Unsupported data type")
    return cls(arr)
