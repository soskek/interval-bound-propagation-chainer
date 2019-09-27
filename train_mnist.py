#!/usr/bin/env python
import argparse
import json

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

import layers
from models import SmallCNN, MediumCNN


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    parser.add_argument('--total-steps', '-t', type=int, default=60001)
    parser.add_argument('--warmup-steps', '-ws', type=int, default=2000)
    parser.add_argument('--rampup-steps', '-rs', type=int, default=10000)
    parser.add_argument('--training-epsilon', '-e', type=float, default=0.3)
    parser.add_argument('--normal-loss-weight', '--kappa', '-k',
                        type=float, default=0.5)
    parser.add_argument('--model-class', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--baseline', action='store_true')

    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print(json.dumps(args.__dict__, indent=2, sort_keys=True))

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # TODO: switch of Verifiable or not
    ModelClass = {
        'small': SmallCNN,
        'medium': MediumCNN,
    }[args.model_class]
    if args.baseline:
        print('Baseline without verification.')
        # model = ModelClass(n_class=10)
        print('Verification is activated!!')
        model = ModelClass(n_class=10,
                           verify=True,
                           warmup_steps=args.total_steps,  # do not rampup
                           rampup_steps=args.total_steps,
                           normal_loss_weight=args.normal_loss_weight,
                           epsilon=0.)
    else:
        print('Verification is activated!!')
        model = ModelClass(n_class=10,
                           verify=True,
                           warmup_steps=args.warmup_steps,
                           rampup_steps=args.rampup_steps,
                           normal_loss_weight=args.normal_loss_weight,
                           epsilon=args.training_epsilon)
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(
        updater, (args.total_steps, 'iteration'), out=args.out)

    # Learning rate decay
    assert args.total_steps >= 25000
    trainer.extend(extensions.MultistepShift(
        'alpha', 0.1, step_value=[15000, 25000], init=None))
    trainer.extend(extensions.observe_value(
        'alpha', lambda trainer: trainer.updater.get_optimizer('main').alpha),
        trigger=(100, 'iteration'))

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator.default_name = 'V'
    trainer.extend(evaluator)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # TODO(niboshi): Temporarily disabled for chainerx. Fix it.
    if device.xp is not chainerx:
        trainer.extend(extensions.DumpGraph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.total_steps if args.frequency == -1 \
        else max(1, args.frequency)
    # trainer.extend(extensions.snapshot(), trigger=(frequency, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(
            model, 'snapshot_model_{.updater.iteration}.npz'),
        trigger=(frequency, 'iteration'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')), trigger=(
        args.total_steps // 100, 'iteration'))

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'V/main/loss',
                 'main/normal', 'V/main/normal',
                 'main/spec', 'V/main/spec',
                 ],
                'iteration', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/acc', 'V/main/acc',
                 'main/vrf_acc',
                 'V/main/vrf_acc',
                 ],
                'iteration', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['iteration', 'main/loss', 'V/main/loss',
         'main/acc', 'V/main/acc',
         'main/normal', 'V/main/normal',
         'main/spec', 'V/main/spec',
         'main/vrf_acc', 'V/main/vrf_acc',
         'main/kappa',
         'main/eps',
         'alpha',
         'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume is not None:
        # Resume from a snapshot
        print('load', args.resume)
        old = model.copy('copy')
        chainer.serializers.load_npz(args.resume, model)
        for (old_n, old_v), (n, v) in zip(old.namedparams(), model.namedparams()):
            print(old_n, n)
            if old_v.array is not None:
                print(old_v.array.flatten()[:2])
            else:
                print('None')
            print(v.array.flatten()[:2])

    # Run the training
    trainer.run()
    print('# of model params', model.count_params())


if __name__ == '__main__':
    main()
