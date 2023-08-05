"""
step 23
"""
from __future__ import annotations

import numpy as np

from src.modules.dezero import Variable


def step23() -> None:
    """
    step 23
    """
    if "__file__" in globals():
        import os
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    x: Variable = Variable(np.array(1.0))
    y: Variable = (x + 3) ** 2
    y.backward()
    print(y)
    print(x.grad)
