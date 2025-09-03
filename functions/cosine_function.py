from . import base_function
import numpy as np

from typing import Tuple


class CosineFunction(base_function.BaseFunction):
    def generate_data(self, range: Tuple[int, int], steps: float) -> Tuple[float, float]:
        x = np.arange(range[0], range[1], steps)
        y = np.cos(x)
        
        data = list(zip(x, y))

        return data
