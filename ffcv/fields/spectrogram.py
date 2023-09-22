from dataclasses import replace
from typing import Optional, Tuple, Type, Callable

import json
import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.compiler import Compiler
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..libffcv import memcpy


class SpectrogramDecoder(Operation):
    def __init__(self, timesteps: int):
        super().__init__()
        self.timesteps = timesteps

        # placeholders
        self.n_elements = None
        self.elems_per_frame = None
        self.itemsize = 4  # TODO: make it robust to different dtypes

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        chans = self.metadata['chans'].max()
        freqs = self.metadata['freqs'].max()

        output_shape = (chans, freqs, self.timesteps)

        self.n_elements = np.prod(output_shape)
        self.elems_per_frame = int(chans * freqs)

        return (
            replace(previous_state, jit_mode=True, shape=output_shape, dtype=self.field.dtype),
            AllocationQuery(output_shape, self.field.dtype)
        )

    def generate_code_v0(self) -> Callable:
        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)
        my_range = Compiler.get_iterator()
        n_frames = self.timesteps
        it_size = self.itemsize * self.elems_per_frame

        def decoder(batch_indices, destination, metadata, storage_state):
            for dest_ix in my_range(batch_indices.shape[0]):
                source_ix = batch_indices[dest_ix]

                # read data in memory
                data = mem_read(metadata[source_ix]['ptr'], storage_state)

                # define random index and crop data
                rd_idx = np.random.randint(data.shape[0] // it_size - n_frames)
                my_memcpy(data[it_size * rd_idx: it_size * (rd_idx + n_frames)],
                          destination[dest_ix])
            return np.transpose(destination, (0, 2, 3, 1))

        return decoder

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)
        my_range = Compiler.get_iterator()
        itemsize = self.itemsize
        elems = self.elems_per_frame
        n_frames = self.timesteps
        it_size = self.itemsize * self.elems_per_frame

        def decoder(batch_indices, destination, metadata, storage_state):
            for dest_ix in my_range(batch_indices.shape[0]):
                source_ix = batch_indices[dest_ix]

                # read data in memory
                data = mem_read(metadata[source_ix]['ptr'], storage_state)
                data = data.reshape(-1, elems, itemsize).transpose(1, 0, 2)

                # define random index and crop data
                rd_idx = 1  #np.random.randint(data.shape[1] - n_frames)
                my_memcpy(data[:, rd_idx: rd_idx + n_frames, :],
                          destination[dest_ix])
            return destination

        return decoder


    def generate_code_v2(self) -> Callable:
        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)
        my_range = Compiler.get_iterator()
        n_bytes = self.n_elements * self.itemsize

        def decoder(batch_indices, destination, metadata, storage_state):
            storage_state[2].fill(n_bytes)
            for dest_ix in my_range(batch_indices.shape[0]):
                source_ix = batch_indices[dest_ix]

                # read data in memory
                data = mem_read(metadata[source_ix]['ptr'], storage_state)
                my_memcpy(data, destination[dest_ix])
            return np.transpose(destination, (0, 2, 3, 1))

        return decoder


SpectrogramArgsType = np.dtype([
    ('shape', '<u8', 3),
    ('type_length', '<u8')
])


class SpectrogramField(Field):
    def __init__(self, dtype: np.dtype = np.float32, num_channels: int = 1):
        self.dtype = dtype
        self.num_channels = 1

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('ptr', '<u8'),
            ('chans', '<u1'),
            ('freqs', '<u8'),
            ('steps', '<u8')
        ])

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return SpectrogramField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        if field.ndim == 4:
            field = field.squeeze(0)
        chans, freqs, steps = field.shape
        ptr, buffer = malloc(field.nbytes)
        buffer[:] = np.transpose(field, axes=(2, 0, 1)).reshape(-1).view('<u1')
        destination['ptr'] = ptr
        destination['chans'] = chans
        destination['freqs'] = freqs
        destination['steps'] = steps

    def get_decoder_class(self) -> Type[Operation]:
        return SpectrogramDecoder
