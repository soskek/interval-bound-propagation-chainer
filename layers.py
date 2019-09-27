#!/usr/bin/env python
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

_TEST_E = 1e-5
_DO_TEST = False


class VerifiableExample(chainer.Chain):  # Example
    def __init__(self, *args, **kwargs):
        super(VerifiableExample, self).__init__(
            *args, **kwargs)

    def forward(self, x):
        y = super(VerifiableExample, self).__call__(x)
        if not(hasattr(x, 'lower') and hasattr(x, 'upper')):
            # return y
            raise NotImplementedError
        # y, lower, upper = ...
        # y.lower = lower
        # y.upper = upper
        # return y
        raise NotImplementedError


def verifiable_relu(x):
    # Standard call
    y = F.relu(x)
    if not(hasattr(x, 'lower') and hasattr(x, 'upper')):
        return y
    # Call with bounds
    y.lower, y.upper = F.relu(x.lower), F.relu(x.upper)
    if _DO_TEST:
        xp = chainer.cuda.get_array_module(x)
        xp.testing.assert_array_less(y.lower.array, y.array + _TEST_E)
        xp.testing.assert_array_less(y.array, y.upper.array + _TEST_E)
    return y


class VerifiableLinear(L.Linear):
    def __init__(self, *args, **kwargs):
        super(VerifiableLinear, self).__init__(
            *args, **kwargs)

    def forward(self, x, n_batch_axes=1):
        if n_batch_axes != 1:
            raise NotImplementedError
        if self.W.array is None:
            in_size = chainer.utils.size_of_shape(x.shape[n_batch_axes:])
            self._initialize_params(in_size)

        # Standard call
        y = super(VerifiableLinear, self).forward(x)
        if not(hasattr(x, 'lower') and hasattr(x, 'upper')):
            return y

        # Call with bounds
        lower, upper = x.lower, x.upper
        c = (lower + upper) / 2.
        r = (upper - lower) / 2.
        c = F.linear(c, self.W, self.b, n_batch_axes=n_batch_axes)
        r = F.linear(r, abs(self.W), b=None, n_batch_axes=n_batch_axes)
        y.lower = c - r
        y.upper = c + r
        if _DO_TEST:
            self.xp.testing.assert_array_less(y.lower.array, y.array + _TEST_E)
            self.xp.testing.assert_array_less(y.array, y.upper.array + _TEST_E)
        return y


class VerifiableConvolution2D(L.Convolution2D):
    def __init__(self, *args, **kwargs):
        super(VerifiableConvolution2D, self).__init__(
            *args, **kwargs)

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])

        # Standard call
        y = super(VerifiableConvolution2D, self).forward(x)
        if not(hasattr(x, 'lower') and hasattr(x, 'upper')):
            return y

        # Call with bounds
        lower, upper = x.lower, x.upper
        c = (lower + upper) / 2.
        r = (upper - lower) / 2.
        c = F.convolution_2d(c, self.W, b=self.b,
                             stride=self.stride, pad=self.pad)
        r = F.convolution_2d(r, abs(self.W), b=None,
                             stride=self.stride, pad=self.pad)
        y.lower = c - r
        y.upper = c + r
        if _DO_TEST:
            self.xp.testing.assert_array_less(y.lower.array, y.array + _TEST_E)
            self.xp.testing.assert_array_less(y.array, y.upper.array + _TEST_E)
        return y
