from base_function import BaseFunction
from sine_function import SineFunction
from cosine_function import CosineFunction
from quadratic_function import QuadraticFunction
from tan_function import TanFunction


class FunctionFactory:
    def __new__(cls):
        raise TypeError("Cannot instantiate FunctionFactory")


    @staticmethod
    def create_function(function_name: str) -> BaseFunction:
        if function_name == 'sine':
            return SineFunction()
        elif function_name == 'cosine':
            return CosineFunction()
        elif function_name == 'quadratic':
            return QuadraticFunction()
        elif function_name == 'tan':
            return TanFunction()

        return None