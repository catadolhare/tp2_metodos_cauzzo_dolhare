"""
Sample code automatically generated on 2024-06-20 22:27:30

by www.matrixcalculus.org

from input

d/db (((tanh(w'*i+b) + 1)/2) - d)^2 = (1-tanh(b+w'*i).^2)*((1+tanh(b+w'*i))/2-d)

where

b is a scalar
d is a scalar
i is a vector
w is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 1
    i_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert i_rows == w_rows

    t_0 = np.tanh((b + (w).dot(i)))
    t_1 = (((1 + t_0) / 2) - d)
    functionValue = (t_1 ** 2)
    gradient = ((1 - (t_0 ** 2)) * t_1)

    return functionValue, gradient

def checkGradient(b, d, i, w):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = float(np.random.randn(1))
    f1, _ = fAndG(b + t * delta, d, i, w)
    f2, _ = fAndG(b - t * delta, d, i, w)
    f, g = fAndG(b, d, i, w)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=0)))

def generateRandomData():
    b = np.random.randn(1)
    d = np.random.randn(1)
    i = np.random.randn(3)
    w = np.random.randn(3)

    return b, d, i, w

if __name__ == '__main__':
    b, d, i, w = generateRandomData()
    functionValue, gradient = fAndG(b, d, i, w)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(b, d, i, w)
