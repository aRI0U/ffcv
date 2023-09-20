from typing import Optional, Tuple

import numpy as np

from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State


class WaveformDecoder(Operation):
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        max_size = self.metadata["size"].max()
        raise NotImplementedError


class WaveformField(Field):
    def __init__(self, dtype: np.dtype = np.float32, num_channels: int = 1):
        self.dtype = dtype



