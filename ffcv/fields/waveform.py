from dataclasses import replace
from typing import Callable, Optional, Tuple, Type

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.compiler import Compiler
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..libffcv import memcpy


class WaveformDecoder(Operation):
    def __init__(self, chunk_size: int, num_channels: int = 1, dtype: str = '<f4'):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.output_shape = (num_channels, chunk_size)
        self.dtype = dtype

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        dtype = np.dtype(self.dtype)
        return (
            replace(previous_state, jit_mode=True, shape=self.output_shape, dtype=dtype),
            AllocationQuery(self.output_shape, dtype)
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)
        my_range = Compiler.get_iterator()

        chans = self.num_channels
        n_frames = self.chunk_size
        dtype = self.dtype
        # print(dtype)
        it_size = int(dtype[-1])

        def decoder(batch_indices, destination, metadata, storage_state):
            for dest_ix in my_range(batch_indices.shape[0]):
                source_ix = batch_indices[dest_ix]

                # read data in memory
                data = mem_read(metadata[source_ix]['ptr'], storage_state)
                data = data.reshape(chans, -1)

                # define random index and crop data
                rd_idx = np.random.randint(data.shape[1] // it_size - n_frames + 1)
                my_memcpy(np.ascontiguousarray(data[:, it_size*rd_idx: it_size*(rd_idx + n_frames)]),
                          destination[dest_ix])

            return destination

        return decoder

class WaveformField(Field):
    def __init__(self, dtype: np.dtype = '<f4', num_channels: int = 1):
        self.dtype = dtype
        self.num_channels = num_channels

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('ptr', '<u8'),
            ('chans', '<u1'),
            ('samples', '<u8')
        ])

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return WaveformField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        chans, samples = field.shape
        ptr, buffer = malloc(field.nbytes)

        buffer[:] = field.reshape(-1).view('<u1')
        destination['ptr'] = ptr
        destination['chans'] = chans
        destination['samples'] = samples

    def get_decoder_class(self) -> Type[Operation]:
        return WaveformDecoder
