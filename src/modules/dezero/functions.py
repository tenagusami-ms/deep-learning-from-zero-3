"""
functions
"""
from __future__ import annotations

import numpy as np

from src.modules.dezero.core import Function, Variable, as_array


class Sin(Function):
    """
    Sin
    """

    def forward(self, x: np.ndarray) -> Variable:
        """
        forward
        """
        return np.sin(x)

    def backward(self, gy: Variable) -> Variable:
        """
        backward
        """
        x, = self.inputs
        return gy * np.cos(as_array(x.data))


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
        return np.cos(x)

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
