from dataclasses import replace
from typing import Optional, Tuple, Type

import json
import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State


class SpectrogramDecoder(Operation):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        # placeholders
        self.max_freqs = None
        self.max_steps = None

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        freqs = self.metadata['freqs']
        steps = self.metadata['steps']
        chans = self.metadata['chans'].max()

        self.max_freqs = np.uint64(freqs.max())
        self.max_steps = np.uint64(steps.max())

        output_shape = (chans, *self.output_size)
        return (
            replace(previous_state, jit_mode=True, shape=output_shape, dtype=self.field.dtype),
            AllocationQuery(np.prod(output_shape), self.field.dtype)
        )


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
        header_size = SpectrogramArgsType.itemsize
        header = binary[:header_size].view(SpectrogramArgsType)[0]
        type_length = header['type_length']
        type_data = binary[header_size:][:type_length].tobytes().decode('ascii')
        type_desc = json.loads(type_data)
        type_desc = [tuple(x) for x in type_desc]
        assert len(type_desc) == 1
        shape = list(header['shape'])
        print('td', type_desc, shape)

        num_channels, num_freqs, num_timesteps = shape[:3]

        return SpectrogramField(num_channels=num_channels)

    def to_binary(self) -> ARG_TYPE:
        return np.ones(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        chans, freqs, steps = field.shape
        ptr, buffer = malloc(np.prod(field.shape))
        buffer[:] = field.reshape(-1)
        destination['ptr'] = ptr
        destination['chans'] = chans
        destination['freqs'] = freqs
        destination['steps'] = steps

    def get_decoder_class(self) -> Type[Operation]:
        return SpectrogramDecoder
