#!/usr/bin/env python
import argparse

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

from layers import verifiable_relu, VerifiableLinear, VerifiableConvolution2D
# , VerifiableLinearWithXEntLoss

# initializer = chainer.initializers.Orthogonal(scale=1.0)
# This doesn't work sometimes
# -> ValueError: Cannot make orthogonal system because # of vectors (32) is larger than that of dimensions (9)
# Seems that k x k should be larger than # of output channels.

initializer = None


class VerifiableClassifier(chainer.Chain):

    def __init__(self, n_class, verify=False,
                 warmup_steps=2000,
                 rampup_steps=10000,
                 normal_loss_weight=0.5,
                 epsilon=0.1):
        super(VerifiableClassifier, self).__init__()
        self.verify = verify
        if self.verify:
            self._Linear = VerifiableLinear
            self._Convolution2D = VerifiableConvolution2D
            self._Activation = verifiable_relu
        else:
            self._Linear = L.Linear
            self._Convolution2D = L.Convolution2D
            self._Activation = F.relu

        # verification-params
        if self.verify:
            self.x_range = [0., 1.]
            self.x_scale = 1.
            self.epsilon = 0.
            self.epsilon_init = 0.
            self.epsilon_last = epsilon
            self.normal_loss_weight = 1.
            self.normal_loss_weight_init = 1.
            self.normal_loss_weight_last = normal_loss_weight
            self.i_update = 0

            assert warmup_steps is not None
            assert rampup_steps is not None
            self.warmup_steps = warmup_steps
            self.rampup_steps = rampup_steps
            print('and loss weight is linearly decayed [{}, {}] between [{}, {}] iterations,'.format(
                self.normal_loss_weight_init, self.normal_loss_weight_last, self.warmup_steps, self.rampup_steps))
            print('and epsilon is linearly increased [{}, {}] between [{}, {}].'.format(
                self.epsilon_init, self.epsilon_last, self.warmup_steps, self.rampup_steps))
        else:
            self.normal_loss_weight = 1.

    def preprocess_for_bounds(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        if self.verify:
            x.lower = F.clip(x - self.epsilon * self.x_scale, *self.x_range)
            x.upper = F.clip(x + self.epsilon * self.x_scale, *self.x_range)
        return x

    def forward(self, x, t):
        """
        # sample
        h = self.l1(x)
        h = self.a1(h)
        h = self.l2(h)
        h = self.a2(h)
        h = self.l3(h)
        h = self.a3(h)
        loss = self.calculate_cross_entropy(h, t=t)
        return loss
        """
        raise NotImplementedError

    def calculate_logit(self, x, t=None, n_batch_axes=1):
        if n_batch_axes != 1:
            raise NotImplementedError
        if self.lo.W.array is None:
            in_size = chainer.utils.size_of_shape(x.shape[n_batch_axes:])
            self.lo._initialize_params(in_size)

        # Standard call
        y = self.lo(x)
        if not(hasattr(x, 'lower') and hasattr(x, 'upper')):
            return y

        # Call with bounds
        if isinstance(t, chainer.Variable):
            t = t.array
        w = self.lo.W
        b = self.lo.b
        batchsize = x.shape[0]
        n_class = b.shape[0]

        w_correct = w[t]  # (batchsize, dim)
        b_correct = b[t]  # (batchsize, )

        _ar2d = self.xp.tile(self.xp.arange(n_class), (batchsize, 1))
        wrong_ids = _ar2d[_ar2d != t[:, None]].reshape(
            (batchsize, n_class - 1))
        w_wrong = w[wrong_ids]  # (batchsize, n_class - 1, dim)
        b_wrong = b[wrong_ids]  # (batchsize, n_class - 1)

        w = w_wrong - w_correct[:, None, :]
        b = b_wrong - b_correct[:, None]
        w = F.transpose(w, (0, 2, 1))  # (batchsize, dim, n_class - 1)

        lower, upper = x.lower, x.upper
        c = (lower + upper) / 2.  # (batchsize, dim)
        r = (upper - lower) / 2.
        c = F.einsum('ij,ijk->ik', c, w)  # (batchsize, n_class - 1)
        if b is not None:
            c += b
        r = F.einsum('ij,ijk->ik', r, abs(w))
        y.worst = c + r
        return y

    def calculate_cross_entropy(self, x, t, n_batch_axes=1):
        logit = self.calculate_logit(x, t=t, n_batch_axes=n_batch_axes)
        normal_loss = F.softmax_cross_entropy(logit, t)
        if hasattr(logit, 'worst'):
            assert self.verify  # TODO: stop this when verifiable evalution for baseline?
            assert logit.worst.shape[1] == logit.shape[1] - 1
            batchsize, n_class = logit.shape
            # concat 0-score for true label
            zeros = self.xp.zeros((batchsize, 1), dtype=x.dtype)
            worst_logit = F.concat([logit.worst, zeros], axis=1)
            label = self.xp.full((batchsize, ),
                                 fill_value=n_class - 1, dtype=t.dtype)
            spec_loss = F.softmax_cross_entropy(worst_logit, label)
            chainer.reporter.report(
                {'vrf_acc': F.accuracy(worst_logit, label)}, self)
        else:
            spec_loss = 0.
            chainer.reporter.report({'vrf_acc': -1.}, self)
        chainer.reporter.report({'normal': normal_loss}, self)
        chainer.reporter.report({'spec': spec_loss}, self)
        chainer.reporter.report({'acc': F.accuracy(logit, t)}, self)

        if chainer.config.train and self.verify:
            self.update_schedule()
        normal_loss = self.normal_loss_weight * normal_loss
        spec_loss = (1 - self.normal_loss_weight) * spec_loss
        loss = normal_loss + spec_loss
        chainer.reporter.report({'loss': loss}, self)
        return loss

    def linear_schedule(self, step, init_step, final_step, init_value, final_value):
        assert final_step >= init_step
        rate = (step - init_step) / \
            float(final_step - init_step)
        linear_value = rate * (final_value - init_value) + init_value
        return np.clip(linear_value,
                       min(init_value, final_value),
                       max(init_value, final_value))

    def update_schedule(self):
        self.i_update += 1.
        self.normal_loss_weight = self.linear_schedule(
            self.i_update, self.warmup_steps, self.warmup_steps + self.rampup_steps,
            self.normal_loss_weight_init, self.normal_loss_weight_last)
        self.epsilon = self.linear_schedule(
            self.i_update, self.warmup_steps, self.warmup_steps + self.rampup_steps,
            self.epsilon_init, self.epsilon_last)
        chainer.reporter.report(
            {'kappa': self.normal_loss_weight}, self)
        chainer.reporter.report(
            {'eps': self.epsilon}, self)


class SmallCNN(VerifiableClassifier):

    def __init__(self, n_class, verify=False,
                 warmup_steps=2000,
                 rampup_steps=10000,
                 normal_loss_weight=0.5,
                 epsilon=0.1):
        super(SmallCNN, self).__init__(
            n_class, verify=verify,
            warmup_steps=warmup_steps, rampup_steps=rampup_steps,
            normal_loss_weight=normal_loss_weight, epsilon=epsilon)

        with self.init_scope():
            self.l1 = self._Convolution2D(
                None, out_channels=16, ksize=(4, 4), stride=2, pad=0,
                initialW=initializer)
            self.a1 = self._Activation
            self.l2 = self._Convolution2D(
                None, out_channels=32, ksize=(4, 4), stride=1, pad=0,
                initialW=initializer)
            self.a2 = self._Activation
            self.l3 = self._Linear(None, 100, initialW=initializer)
            self.a3 = self._Activation
            self.lo = self._Linear(None, n_class, initialW=initializer)

    def forward(self, x, t):
        # TODO: no t prediction
        x = self.preprocess_for_bounds(x)
        h = self.l1(x)
        h = self.a1(h)
        h = self.l2(h)
        h = self.a2(h)
        h = self.l3(h)
        h = self.a3(h)
        loss = self.calculate_cross_entropy(h, t=t)
        return loss


class MediumCNN(VerifiableClassifier):

    def __init__(self, n_class, verify=False,
                 warmup_steps=2000,
                 rampup_steps=10000,
                 normal_loss_weight=0.5,
                 epsilon=0.1):
        super(MediumCNN, self).__init__(
            n_class, verify=verify,
            warmup_steps=warmup_steps, rampup_steps=rampup_steps,
            normal_loss_weight=normal_loss_weight, epsilon=epsilon)

        with self.init_scope():
            self.l1 = self._Convolution2D(
                None, out_channels=32, ksize=(3, 3), stride=1, pad=0,
                initialW=initializer)
            self.a1 = self._Activation
            self.l2 = self._Convolution2D(
                None, out_channels=32, ksize=(4, 4), stride=2, pad=0,
                initialW=initializer)
            self.a2 = self._Activation
            self.l3 = self._Convolution2D(
                None, out_channels=64, ksize=(3, 3), stride=1, pad=0,
                initialW=initializer)
            self.a3 = self._Activation
            self.l4 = self._Convolution2D(
                None, out_channels=64, ksize=(4, 4), stride=2, pad=0,
                initialW=initializer)
            self.a4 = self._Activation
            self.l5 = self._Linear(None, 512, initialW=initializer)
            self.a5 = self._Activation
            self.l6 = self._Linear(None, 512, initialW=initializer)
            self.a6 = self._Activation
            self.lo = self._Linear(None, n_class, initialW=initializer)

    def forward(self, x, t):
        # TODO: no t prediction
        x = self.preprocess_for_bounds(x)
        h = self.l1(x)
        h = self.a1(h)
        h = self.l2(h)
        h = self.a2(h)
        h = self.l3(h)
        h = self.a3(h)
        h = self.l4(h)
        h = self.a4(h)
        h = self.l5(h)
        h = self.a5(h)
        h = self.l6(h)
        h = self.a6(h)
        loss = self.calculate_cross_entropy(h, t=t)
        return loss
