"""
step 14
"""
from __future__ import annotations

from typing import Optional, Sequence, MutableSequence

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

    def backward(self) -> None:
        """
        backward
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: MutableSequence[Optional[Function]] = [self.creator]
        while funcs:
            func: Optional[Function] = funcs.pop()
            gys: Sequence[Optional[np.ndarray]] = [output.grad for output in func.outputs]
            gxs: tuple[np.ndarray, ...] | np.ndarray = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs: tuple[np.ndarray, ...] = (gxs,)
            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)

    def clear_grad(self) -> None:
        """
        clear_grad
        """
        self.grad = None


class Function:
    """
    Function
    """

    def __call__(self, *inputs: Variable) -> Sequence[Variable] | Variable:
        """
        __call__
        """
        xs: Sequence[np.ndarray] = [x.data for x in inputs]
        ys: tuple[np.ndarray, ...] | np.ndarray = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys: tuple[np.ndarray, ...] = (ys,)
        outputs: Sequence[Variable] = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs: Sequence[Variable] = inputs
        self.outputs: Sequence[Variable] = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        raise NotImplementedError()


class Add(Function):
    """
    Add
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        return x0 + x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        return gy, gy


class Square(Function):
    """
    Square
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward
        """
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        backward
        """
        x: np.ndarray = self.inputs[0].data
        gx: np.ndarray = 2 * x * gy
        return gx


def step14() -> None:
    """
    step 14
    """
    x: Variable = Variable(np.array(3.0))
    y: Variable = add(x, x)
    y.backward()
    print(x.grad)

    x.clear_grad()
    y: Variable = add(add(x, x), x)
    y.backward()
    print(x.grad)


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


def square(x: Variable) -> Variable:
    """
    square
    """
    return Square()(x)
