"""
Useful audio transforms
"""
import numpy as np
import torch as ch
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler


def ch_dtype_from_numpy(dtype):
    return ch.from_numpy(np.zeros((), dtype=dtype)).dtype


class Int16ToFloat(Operation):
    """Fast implementation of normalization and type conversion for int16 audios to any floating point dtype.

    Works on both GPU and CPU tensors.

    Parameters
    ----------
    type: np.dtype
        The desired output type for the result as a numpy type.
        If the transform is applied on a GPU tensor it will be converted
        as the equivalent torch dtype.
    """

    def __init__(self, dtype: np.dtype = np.float32):
        super().__init__()
        num_values = 2 ** 15
        table = np.arange(num_values)
        self.original_dtype = dtype
        table = table.astype(dtype)
        table = table / num_values  # normalize floats between -1. and 1.
        if dtype == np.float16:  # TODO: understand that shit
            dtype = np.int16
            table = table.view(dtype)

        self.dtype = dtype

        self.lookup_table = table
        self.previous_shape = None
        self.mode = 'cpu'

    def generate_code(self) -> Callable:
        if self.mode == 'cpu':
            return self.generate_code_cpu()
        return self.generate_code_gpu()

    def generate_code_gpu(self) -> Callable:

        # We only import cupy if it's truly needed
        import cupy as cp
        import pytorch_pfn_extras as ppe

        tn = np.zeros((), dtype=self.dtype).dtype.name

        # since PyTorch does not support uint16, we have to deal with negative indices in the CUDA code directly
        kernel = cp.ElementwiseKernel(f'int16 input, raw {tn} table', f'{tn} output', 'output = input > 0 ? table[input] : -table[-input];')
        final_type = ch_dtype_from_numpy(self.original_dtype)

        def normalize_convert(audios, result):
            B, C, N = audios.shape

            table = self.lookup_table

            result = result[:B]
            result_c = result.view(-1)
            audios = audios.view(-1)

            current_stream = ch.cuda.current_stream()
            with ppe.cuda.stream(current_stream):
                kernel(audios, table, result_c)

            # Mark the result as channel last
            final_result = result.reshape(B, C, N)

            return final_result.view(final_type)

        return normalize_convert

    def generate_code_cpu(self) -> Callable:

        table = self.lookup_table.view(dtype=self.dtype)
        my_range = Compiler.get_iterator()

        def normalize_convert(audios, result, indices):
            result_flat = result.reshape(result.shape[0], -1)
            num_samples = result_flat.shape[1]
            for i in my_range(len(indices)):
                audio = audios[i].reshape(num_samples).view(np.uint16)
                for s in range(num_samples):
                    # Just in case llvm forgets to unroll this one
                    result_flat[i, s] = table[audio[s]]

            return result

        normalize_convert.is_parallel = True
        normalize_convert.with_indices = True
        return normalize_convert

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:

        if previous_state.device == ch.device('cpu'):
            new_state = replace(previous_state, jit_mode=True, dtype=self.dtype)
            return new_state, AllocationQuery(
                shape=previous_state.shape,
                dtype=self.dtype,
                device=previous_state.device
            )

        else:
            self.mode = 'gpu'
            new_state = replace(previous_state, dtype=self.dtype)

            gpu_type = ch_dtype_from_numpy(self.dtype)

            # Copy the lookup table into the proper device
            try:
                self.lookup_table = ch.from_numpy(self.lookup_table)
            except TypeError:
                pass  # This is alredy a tensor
            self.lookup_table = self.lookup_table.to(previous_state.device)

            return new_state, AllocationQuery(
                shape=previous_state.shape,
                device=previous_state.device,
                dtype=gpu_type
            )