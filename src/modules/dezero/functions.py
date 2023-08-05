"""
functions
"""
from __future__ import annotations

import numpy as np

from src.modules.dezero.core import Function, Variable, as_variable, as_array


class Sin(Function):
    """
    Sin
    """

    def forward(self, x: np.ndarray) -> Variable:
        """
        forward
        """
        return as_variable(as_array(np.sin(x)))

    def backward(self, gy: Variable) -> Variable:
        """
        backward
        """
        x, = self.inputs
        return gy * np.cos(x)


def sine(x: Variable) -> Variable:
    """
    sine
    """
    return Sin()(x)


class Cos(Function):
    """
    Sin
    """

    def forward(self, x: np.ndarray) -> Variable:
        """
        forward
        """
        return as_variable(as_array(np.cos(x)))

    def backward(self, gy: Variable) -> Variable:
        """
        backward
        """
        x, = self.inputs
        return gy * (-sine(x))


def cosine(x: Variable) -> Variable:
    """
    cosine
    """
    return Cos()(x)
