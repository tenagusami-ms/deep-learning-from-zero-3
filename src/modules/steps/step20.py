"""
step 20
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
    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data: np.ndarray = data
        self.name: Optional[str] = name
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def __mul__(self, other: Variable) -> Variable:
        return mul(self, other)

    def __add__(self, other: Variable) -> Variable:
        return add(self, other)

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

    @property
    def shape(self) -> tuple[int, ...]:
        """
        shape
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """
        ndim
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        """
        size
        """
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """
        dtype
        """
        return self.data.dtype


def add(x0: Variable, x1: Variable) -> Variable:
    """
    addition
    """
    return Add()(x0, x1)


def mul(x0: Variable, x1: Variable) -> Variable:
    """
    multiplication
    """
    return Mul()(x0, x1)


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


class Mul(Function):
    """
    Mul
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        return x0 * x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


def step20() -> None:
    """
    step 20
    """
    a: Variable = Variable(np.array(3.0))
    b: Variable = Variable(np.array(2.0))
    c: Variable = Variable(np.array(1.0))

    y: Variable = a * b + c
    y.backward()
    print(y)
    print(a.grad)
    print(b.grad)


def as_array(x: np.ndarray) -> np.ndarray:
    """
    as_array
    """
    if np.isscalar(x):
        return np.array(x)
    return x


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
