import base_function
import numpy as np

from typing import Tuple


class QuadraticFunction(base_function.BaseFunction):
    def generate_data(self, range: Tuple[int, int], steps: float) -> Tuple[float, float]:
        x = np.arange(range[0], range[1], steps)
        y = np.pow(x, 2)
        
        data = list(zip(x, y))

        return data
