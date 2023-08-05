"""
step 34
"""
from __future__ import annotations

import numpy as np

from src.modules.dezero import Variable
from src.modules.dezero.functions import sine

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def step34() -> None:
    """
    step 34
    """
    x: Variable = Variable(np.array(1.0))
    y: Variable = sine(x)
    y.backward(create_graph=True)
    iterations: int = 10

    for i in range(3):
        gx: Variable = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)
        print(x.grad)
