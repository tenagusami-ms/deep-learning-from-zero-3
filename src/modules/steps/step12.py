"""
step 12
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

    def __call__(self, *inputs: Variable) -> Sequence[Variable] | Variable:
        """
        __call__
        """
        xs: Sequence[np.ndarray] = [x.data for x in inputs]
        ys: tuple[np.ndarray] | np.ndarray = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys: tuple[np.ndarray] = (ys,)
        outputs: Sequence[Variable] = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs: Sequence[Variable] = inputs
        self.outputs: Sequence[Variable] = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray] | np.ndarray:
        """
        forward
        """
        raise NotImplementedError()


class Add(Function):
    """
    Add
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray] | np.ndarray:
        """
        forward
        """
        return x0 + x1


def step12() -> None:
    """
    step 12
    """
    x0, x1 = [Variable(np.array(2)), Variable(np.array(3))]
    y: Sequence[Variable] | Variable = add(x0, x1)
    print(y.data)


def as_array(x: np.ndarray) -> np.ndarray:
    """
    as_array
    """
    if np.isscalar(x):
        return np.array(x)
    return x


def add(x0: Variable, x1: Variable) -> Variable:
    """
    add
    """
    return Add()(x0, x1)
