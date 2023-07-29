"""
step 09
"""
from __future__ import annotations

from typing import Optional, MutableSequence

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
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
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: MutableSequence[Optional[Function]] = [self.creator]
        while funcs:
            func: Optional[Function] = funcs.pop()
            x: Variable = func.input
            y: Variable = func.output
            x.grad = func.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


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
        output: Variable = Variable(as_array(y))
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


def square(x: Variable) -> Variable:
    """
    square
    """
    return Square()(x)


def exp(x: Variable) -> Variable:
    """
    exp
    """
    return Exp()(x)


def as_array(x: np.ndarray) -> np.ndarray:
    """
    as_array
    """
    if np.isscalar(x):
        return np.array(x)
    return x


def step09() -> None:
    """
    step 09
    """
    x: Variable = Variable(np.array(0.5))
    y: Variable = square(exp(square(x)))
    y.backward()
    print(x.grad)
