"""
step 24
"""
from __future__ import annotations

import numpy as np

from src.modules.dezero import Variable

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def step24() -> None:
    """
    step 24
    """
    # x: Variable = Variable(np.array(1.0))
    # y: Variable = Variable(np.array(1.0))
    # z: Variable = sphere(x, y)
    # z.backward()
    # print(z)
    # print(x.grad, y.grad)

    # x: Variable = Variable(np.array(1.0))
    # y: Variable = Variable(np.array(1.0))
    # z: Variable = matyas(x, y)
    # z.backward()
    # print(z)
    # print(x.grad, y.grad)

    x: Variable = Variable(np.array(1.0))
    y: Variable = Variable(np.array(1.0))
    z: Variable = goldstein(x, y)
    z.backward()
    print(z)
    print(x.grad, y.grad)


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
