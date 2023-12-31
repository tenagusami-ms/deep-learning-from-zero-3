"""
step 11
"""
from __future__ import annotations

from typing import Optional, Sequence

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


class Function:
    """
    Function
    """

    def __call__(self, inputs: Sequence[Variable]) -> Sequence[Variable]:
        """
        __call__
        """
        xs: Sequence[np.ndarray] = [x.data for x in inputs]
        ys: Sequence[np.ndarray] = self.forward(xs)
        outputs: Sequence[Variable] = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs: Sequence[Variable] = inputs
        self.outputs: Sequence[Variable] = outputs
        return outputs

    def forward(self, xs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        forward
        """
        raise NotImplementedError()


class Add(Function):
    """
    Add
    """

    def forward(self, xs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        forward
        """
        x0: np.ndarray = xs[0]
        x1: np.ndarray = xs[1]
        y: np.ndarray = x0 + x1
        return (y,)


def step11() -> None:
    """
    step 11
    """
    xs: Sequence[Variable] = [Variable(np.array(2)), Variable(np.array(3))]
    f: Function = Add()
    ys: Sequence[Variable] = f(xs)
    print(ys[0].data)


def as_array(x: np.ndarray) -> np.ndarray:
    """
    as_array
    """
    if np.isscalar(x):
        return np.array(x)
    return x
