from __future__ import annotations
import math
import random

from mcl.machine_types import i32, intp, memref
from mcl.dialects import LoopNestAPI
from mcl.ndarray import DType, Array
from mcl.vm import _get_machine_value

def random_array(shape: tuple[intp, ...], dtype: DType, start: int = 0, stop: int = 0) -> Array[T]:
    # TODO: Need to get i32 from array dtype
    data = memref.alloc(shape, i32)
    array = Array(dtype=dtype, data=data)

    for idx in LoopNestAPI.from_tuple(shape):
        array[idx] = i32(random.randint(start, stop))

    return array

def array_exp(array: Array):
    # TODO: Need to get i32 from array dtype
    # TODO: Need to add support for math.exp at scalar level within Int32, shouldn't need to call .value or 
    # _get_machine_value here it should be done within vm through Int32
    # TODO: Need to add support for float32
    
    for idx in LoopNestAPI.from_tuple(array.shape):
        array[idx] = i32(int(math.exp(_get_machine_value(array[idx].value))))

    return array

def array_sqrt(array: Array):
    # TODO: Need to get i32 from array dtype
    # TODO: Need to add support for math.exp at scalar level within Int32, shouldn't need to call .value or 
    # _get_machine_value here it should be done within vm through Int32
    # TODO: Need to add support for float32
    
    for idx in LoopNestAPI.from_tuple(array.shape):
        array[idx] = i32(int(math.sqrt(_get_machine_value(array[idx].value))))

    return array

def array_sum(array: Array, axis: int = -1, keepdims: bool = False):
    # TODO: Support axis and keepdims
    res = i32(0)

    for idx in LoopNestAPI.from_tuple(array.shape):
        res += array[idx].value

    return res

def array_max(array: Array, axis: int = -1, keepdims: bool = False):
    # TODO: Support axis and keepdims
    max = i32(0)

    for idx in LoopNestAPI.from_tuple(array.shape):
        if array[idx] > max:
            max = array[idx]
    
    return max

def array_matmul(matrix_1: Array, matrix_2: Array):
    assert matrix_1.shape[1] == matrix_2.shape[0]

    raise NotImplementedError

