from functions.base_function import BaseFunction
from functions.sine_function import SineFunction
from functions.cosine_function import CosineFunction
from functions.quadratic_function import QuadraticFunction
from functions.tan_function import TanFunction


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