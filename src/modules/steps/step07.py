"""
step 07
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func: Function) -> None:
        """
        set_creator
        """
        self.creator = func

    def backward(self) -> None:
        """
        backward
        """
        func: Optional[Function] = self.creator
        if func is not None:
            x: Variable = func.input
            x.grad = func.backward(self.grad)
            x.backward()


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
        output.set_creator(self)
        self.input: Variable = input_variable
        self.output: Variable = output
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


def step07() -> None:
    """
    step 07
    """
    a_func: Function = Square()
    b_func: Function = Exp()
    c_func: Function = Square()

    x: Variable = Variable(np.array(0.5))
    a: Variable = a_func(x)
    b: Variable = b_func(a)
    y: Variable = c_func(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
