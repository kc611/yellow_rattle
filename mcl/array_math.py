from __future__ import annotations
import math
import random

from mcl.machine_types import i32, intp, memref, f32
from mcl.dialects import LoopNestAPI
from mcl.ndarray import DType, Array, Int32
from mcl.vm import _get_machine_value

def random_array(shape: tuple[intp, ...], dtype: DType, start: int = 0, stop: int = 0) -> Array[T]:
    # TODO: Need to get f32 from array dtype
    data = memref.alloc(shape, f32)
    array = Array(dtype=dtype, data=data)

    for idx in LoopNestAPI.from_tuple(shape):
        array[idx] = f32(random.random() * (stop - start) + start)

    return array

def array_exp(array: Array):
    # TODO: Need to get i32 from array dtype
    # TODO: Need to add support for math.exp at scalar level within Int32, shouldn't need to call .value or 
    # _get_machine_value here it should be done within vm through Int32
    # TODO: Need to add support for float32
    
    for idx in LoopNestAPI.from_tuple(array.shape):
        array[idx] = f32(math.exp(_get_machine_value(array[idx].value)))

    return array

def array_sqrt(array: Array):
    # TODO: Need to get i32 from array dtype
    # TODO: Need to add support for math.exp at scalar level within Int32, shouldn't need to call .value or 
    # _get_machine_value here it should be done within vm through Int32
    # TODO: Need to add support for float32
    
    for idx in LoopNestAPI.from_tuple(array.shape):
        array[idx] = f32(math.sqrt(_get_machine_value(array[idx].value)))

    return array

def array_sum(array: Array, axis: int = -1, keepdims: bool = False):
    # TODO: Support axis and keepdims
    res = f32(0)

    for idx in LoopNestAPI.from_tuple(array.shape):
        res += array[idx].value

    return res

def array_max(array: Array, axis: int = -1, keepdims: bool = False):
    # TODO: Support axis and keepdims
    max = f32(0)

    for idx in LoopNestAPI.from_tuple(array.shape):
        if array[idx].value > max:
            max = array[idx].value
    
    return max

def array_matmul(matrix_1: Array, matrix_2: Array):
    if len(matrix_1.shape) > 2 or len(matrix_2.shape) > 2:
        for i in range(len(matrix_1.shape) - 2):
            assert matrix_1.shape[i] == matrix_2.shape[i]
    assert matrix_1.shape[-1] == matrix_2.shape[-2]

    result_shape = matrix_1.shape[:-1] + matrix_2.shape[-1:]
    result = memref.alloc(result_shape, i32)

    result_array = Array(dtype=matrix_1.dtype, data=result)

    for idx in LoopNestAPI.from_tuple(result_shape):
        result_array[idx] = i32(0)
        for i in range(matrix_1.shape[-1]):
            for j in range(matrix_2.shape[-2]):
                result_array[idx] = result_array[idx].value + matrix_1[idx[:-1] + (i,)].value * matrix_2[idx[:-2] + (j, i)].value

    return result_array
