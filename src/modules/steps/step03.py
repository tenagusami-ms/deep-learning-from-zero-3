"""
step 03
"""
from __future__ import annotations

import numpy as np

from src.modules.steps.step01 import Variable
from src.modules.steps.step02 import Function, Square


class Exp(Function):
    """
    Exp
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
        """
        return np.exp(x)


def step03() -> None:
    """
    step 03
    """
    a_func: Function = Square()
    b_func: Function = Exp()
    c_func: Function = Square()

    x: Variable = Variable(np.array(0.5))
    a: Variable = a_func(x)
    b: Variable = b_func(a)
    y: Variable = c_func(b)
    print(y.data)
