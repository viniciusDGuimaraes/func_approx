import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BaseFunction(ABC):
    @abstractmethod
    def generate_data(self, range: Tuple[int, int], steps: float) -> Tuple[float, float]:
        pass