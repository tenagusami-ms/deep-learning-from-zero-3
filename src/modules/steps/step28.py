"""
step 28
"""
from __future__ import annotations

import math
import numbers

import numpy as np

from src.modules.dezero import Variable, Function

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def step28() -> None:
    """
    step 28
    """
    x0: Variable = Variable(np.array(0.0))
    x1: Variable = Variable(np.array(2.0))

    learning_rate: float = 0.001
    iterations: int = 50000

    for i in range(iterations):
        print(x0, x1)

        y: Variable = rosenbrock(x0, x1)

        x0.clear_grad()
        x1.clear_grad()
        y.backward()

        x0.data -= learning_rate * x0.grad
        x1.data -= learning_rate * x1.grad

    # x.name = "x"
    # y.name = "y"
    #
    # plot_dot_graph(y, verbose=False, to_file="sine.png")


def sphere(x: Variable, y: Variable) -> Variable:
    """
    sphere
    """
    return x ** 2 + y ** 2


def matyas(x: Variable, y: Variable) -> Variable:
    """
    matyas
    """
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x: Variable, y: Variable) -> Variable:
    """
    Goldstein-Price function
    """
    return (
            (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))
            * (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)))


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    """
    Rosenbrock function
    """
    return 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2


class Sin(Function):
    """
    Sin
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
        """
        return np.sin(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        backward
        """
        x: np.ndarray = self.inputs[0].data
        return gy * np.cos(x)


def sine(x: Variable) -> Variable:
    """
    sine
    """
    return Sin()(x)


def my_sine(x: Variable, threshold=0.0001) -> Variable:
    """
    my_sine
    """
    y: Variable | np.ndarray | numbers.Real = 0
    for i in range(100000):
        c: Variable = (-1) ** i / math.factorial(2 * i + 1)
        t: Variable = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y