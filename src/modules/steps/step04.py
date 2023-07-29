"""
step 04
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.modules.steps.step01 import Variable
from src.modules.steps.step02 import Function, Square
from src.modules.steps.step03 import Exp


def numerical_diff(function: Callable, x: Variable, eps: float = 1e-4) -> np.ndarray:
    """
    numerical_diff
    """
    x0: Variable = Variable(x.data - eps)
    x1: Variable = Variable(x.data + eps)
    y0: Variable = function(x0)
    y1: Variable = function(x1)
    return (y1.data - y0.data) / (2 * eps)


def step04() -> None:
    """
    step 04
    """
    f1: Square = Square()
    x1: Variable = Variable(np.array(2.0))
    dy1: np.ndarray = numerical_diff(f1, x1)
    print(dy1)

    x: Variable = Variable(np.array(0.5))
    dy: np.ndarray = numerical_diff(f, x)
    print(dy)


def f(x):
    a_func: Function = Square()
    b_func: Function = Exp()
    c_func: Function = Square()
    return c_func(b_func(a_func(x)))
