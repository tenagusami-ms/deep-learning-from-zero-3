"""
step 10
"""
from __future__ import annotations

import unittest
from typing import Callable

import numpy as np

from step09 import square, Variable


class SquareTest(unittest.TestCase):
    def test_forward(self) -> None:
        """
        test_forward
        """
        x: Variable = Variable(np.array(2.0))
        y: Variable = square(x)
        expected: np.ndarray = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        """
        test_backward
        """
        x: Variable = Variable(np.array(3.0))
        y: Variable = square(x)
        y.backward()
        expected: np.ndarray = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        """
        test_gradient_check
        """
        x: Variable = Variable(np.random.rand(1))
        y: Variable = square(x)
        y.backward()
        num_grad: np.ndarray = numerical_diff(square, x)
        flag: bool = np.allclose(x.grad, num_grad)
        self.assertTrue(flag)


def numerical_diff(function: Callable, x: Variable, eps: float = 1e-4) -> np.ndarray:
    """
    numerical_diff
    """
    x0: Variable = Variable(x.data - eps)
    x1: Variable = Variable(x.data + eps)
    y0: Variable = function(x0)
    y1: Variable = function(x1)
    return (y1.data - y0.data) / (2 * eps)


unittest.main()
