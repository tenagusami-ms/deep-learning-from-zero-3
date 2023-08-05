"""
step 21
"""
from __future__ import annotations

import contextlib
import numbers
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
    __array_priority__ = 200

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

    def __mul__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return mul(self, other)

    def __rmul__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return mul(self, other)

    def __add__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return add(self, other)

    def __radd__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return add(self, other)

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return sub(self, other)

    def __rsub__(self, other: Variable | np.ndarray | numbers.Real):
        return rsub(self, other)

    def __truediv__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return div(self, other)

    def __rtruediv__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return rdiv(self, other)

    def __pow__(self, other: Variable | np.ndarray | numbers.Real) -> Variable:
        return power(self, other)

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


def add(x0: Variable, x1: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    addition
    """
    x1: np.ndarray = as_array(x1)
    return Add()(x0, x1)


def neg(x: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    negation
    """
    x: np.ndarray = as_array(x)
    return Neg()(x)


def sub(x0: Variable, x1: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    subtraction
    """
    x1: np.ndarray = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    subtraction
    """
    x1: np.ndarray = as_array(x1)
    return Sub()(x1, x0)


def mul(x0: Variable, x1: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    multiplication
    """
    x1: np.ndarray = as_array(x1)
    return Mul()(x0, x1)


def div(x0: Variable, x1: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    division
    """
    x1: np.ndarray = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: Variable | np.ndarray | numbers.Real) -> Variable:
    """
    division
    """
    x1: np.ndarray = as_array(x1)
    return Div()(x1, x0)


def power(x: Variable, c: numbers.Real) -> Variable:
    """
    power
    """
    return Pow(c)(x)


class Function:
    """
    Function
    """

    def __call__(self, *inputs: Variable | np.ndarray) -> Sequence[Variable] | Variable:
        """
        __call__
        """
        inputs: Sequence[Variable] = [as_variable(x) for x in inputs]
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


class Neg(Function):
    """
    negate
    """

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        return -x

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        return -gy


class Sub(Function):
    """
    Sub
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        return x0 - x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        return gy, -gy


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


class Div(Function):
    """
    Div
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        return x0 / x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy / x1, gy * (-x0 / x1 ** 2)


class Pow(Function):
    """
    Pow
    """

    def __init__(self, c: numbers.Real) -> None:
        self.c: numbers.Real = c

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        forward
        """
        return x ** self.c

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        backward
        """
        x: np.ndarray = self.inputs[0].data
        c: numbers.Real = self.c
        return c * x ** (c - 1) * gy


def as_array(x: np.ndarray | numbers.Real) -> np.ndarray:
    """
    as_array
    """
    if np.isscalar(x):
        return np.array(x)
    return x


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


def as_variable(obj: np.ndarray | Variable) -> Variable:
    """
    as_variable
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
