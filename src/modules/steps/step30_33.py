"""
step 30 - 33
"""
from __future__ import annotations

import numpy as np

from src.modules.dezero import Variable

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def step30_33() -> None:
    """
    step 30 - 33
    """
    x: Variable = Variable(np.array(2.0))
    iterations: int = 10

    for i in range(iterations):
        print(i, x)

        y: Variable = f(x)
        x.clear_grad()
        y.backward(create_graph=True)

        gx: Variable = x.grad
        x.clear_grad()
        gx.backward()
        gx2: Variable = x.grad

        x.data -= gx.data / gx2.data

    # x: Variable = Variable(np.array(2.0))
    # y: Variable = f(x)
    # y.backward(create_graph=True)
    # print(x.grad)
    #
    # gx: Variable = x.grad
    # x.clear_grad()
    # gx.backward()
    # print(x.grad)

    # x.name = "x"
    # y.name = "y"
    #
    # plot_dot_graph(y, verbose=False, to_file="sine.png")


def f(x: Variable) -> Variable:
    """
    f
    """
    return x ** 4 - 2 * x ** 2
