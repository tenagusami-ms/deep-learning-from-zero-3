"""
step 02
"""
from __future__ import annotations

import numpy as np

from src.modules.steps.step01 import Variable


class Function:
    """
    Function
    """

    def __call__(self, variable: Variable) -> Variable:
        """
        __call__
        """
        x: np.ndarray = variable.data
        y: np.ndarray = self.forward(x)
        output: Variable = Variable(y)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
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


def step02() -> None:
    """
    step 02
    """
    x: Variable = Variable(np.array(10))
    func: Square = Square()
    y: Variable = func(x)
    print(type(y))
    print(y.data)
