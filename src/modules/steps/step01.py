"""
step 01
"""
from __future__ import annotations

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


def step01():
    """
    step 01
    """
    data = Variable(np.array(1.0))
    print(data.data)
    data.data = np.array(2.0)
    print(data.data)
