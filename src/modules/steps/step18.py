"""
step 18
"""
from __future__ import annotations

import contextlib
import weakref
from _weakref import ReferenceType
from typing import Optional, Sequence, MutableSet

import numpy as np


class Config:
    """
    Config
    """
    enable_backprop: bool = True


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    def set_creator(self, func: Function) -> None:
        """
        set_creator
        """
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool = False) -> None:
        """
        backward
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = list()
        seen_set: MutableSet[Function] = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda f_in_funcs: f_in_funcs.generation)

        add_func(self.creator)
        while funcs:
            func: Optional[Function] = funcs.pop()
            gys: Sequence[Optional[np.ndarray]] = [output().grad for output in func.outputs]
            gxs: tuple[np.ndarray, ...] | np.ndarray = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs: tuple[np.ndarray, ...] = (gxs,)
            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in func.outputs:
                    y().grad = None

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

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs: Sequence[Variable] = inputs
            self.outputs: Sequence[ReferenceType[Variable]] = [weakref.ref(output) for output in outputs]
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


def step18() -> None:
    """
    step 18
    """
    # x0: Variable = Variable(np.array(1.0))
    # x1: Variable = Variable(np.array(1.0))
    # t: Variable = add(x0, x1)
    # y: Variable = add(x0, t)
    # y.backward()
    # print(y.grad, t.grad)
    # print(x0.grad, x1.grad)
    #
    # Config.enable_backprop = True
    # x: Variable = Variable(np.ones((100, 100, 100)))
    # y: Variable = square(square(square(x)))
    # y.backward()
    #
    # Config.enable_backprop = False
    # x: Variable = Variable(np.ones((100, 100, 100)))
    # y: Variable = square(square(square(x)))

    with no_grad():
        x: Variable = Variable(np.array(2.0))
        y: Variable = square(x)
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


def square(x: Variable) -> Variable:
    """
    square
    """
    return Square()(x)


@contextlib.contextmanager
def using_config(name: str, value: bool) -> None:
    """
    using_config
    """
    old_value: bool = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """
    no_grad
    """
    return using_config("enable_backprop", False)
