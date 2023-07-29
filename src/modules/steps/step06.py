"""
step 05
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None


class Function:
    """
    Function
    """

    def __call__(self, input_variable: Variable) -> Variable:
        """
        __call__
        """
        x: np.ndarray = input_variable.data
        y: np.ndarray = self.forward(x)
        output: Variable = Variable(y)
        self.input = input_variable
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
        """
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        backward
        """
        raise NotImplementedError()


class Square(Function):
    """
    Square
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
        """
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        backward
        """
        x: np.ndarray = self.input.data
        gx: np.ndarray = 2 * x * gy
        return gx


class Exp(Function):
    """
    Exp
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
        """
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        backward
        """
        x: np.ndarray = self.input.data
        gx: np.ndarray = np.exp(x) * gy
        return gx


def step06() -> None:
    """
    step 06
    """
    a_func: Function = Square()
    b_func: Function = Exp()
    c_func: Function = Square()

    x: Variable = Variable(np.array(0.5))
    a: Variable = a_func(x)
    b: Variable = b_func(a)
    y: Variable = c_func(b)

    y.grad = np.array(1.0)
    b.grad = c_func.backward(y.grad)
    a.grad = b_func.backward(b.grad)
    x.grad = a_func.backward(a.grad)

    print(x.grad)
