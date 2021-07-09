# This file is part of ReACORN, a reimplementation by Élie Michel of the ACORN
# paper by Martel et al. published at SIGGRAPH 2021.
#
# Copyright (c) 2021 -- Télécom Paris (Élie Michel <elie.michel@telecom-paris.fr>)
# 
# The MIT license:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided “as is”, without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and non-infringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings
# in the Software.

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from .samplers import PixelAlignedSampler
from .domain_tree import DomainTree
from .profiling import Timer, profiling_counters
from .utils import period_starts, period_ends, ensure_parent_dir
from .figures import TrainingFigure

# -----------------------------------------------------------------------------

class TrainingCallback:
    """
    Function called every 'n' epochs, where n is called the "period" of the
    callback. This could simply be a tuple (n, callback) but we also add some
    additional options to pass extra arguments and tell whether the callback
    must be called rather at the first or last epoch of the period.

    A negative period means that the callback will never get called.

    When used by the Trainer class, the first argument passed to the callback
    is the trainer instance.
    """
    def __init__(self, callback, period=-1, at_period_start=True, *args, **kwargs):
        """
        @param callback function that will be called at each period
        @param period number of epochs between two calls to the callback
        @param at_period_start if True, callback is called when the period
                               starts (at epochs 0, period, 2*period, etc.),
                               otherwise when it ends (at epochs period-1,
                               2*period-1, etc.).
        Extra arguments are forwarded to the callback.
        """
        self.callback = callback
        self.period = period
        self.period_matches = period_starts if at_period_start else period_ends
        self.extra_args = args
        self.extra_kwargs = kwargs

    def maybe_call(self, epoch, *args, **kwargs):
        """
        Call the callback is epoch matches the period pattern. The callback is
        called with extra argument passed to this method, to which are appended
        the extra arguments from the constructor
        """
        if self.period_matches(epoch, self.period):
            all_args = (*args, *self.extra_args)
            all_kwargs = {**kwargs, **self.extra_kwargs}
            self.callback(*all_args, **all_kwargs)


# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self,
        acorn,
        target_image,
        learning_rate=1e-3,
        minimum_block_pixel_size=1,
        train_points_per_block=None,
        sampler=None,
        eval_on_cpu=False):
        """
        Initialize the trainer to operate on a given Acorn model.
        
        @param acorn Acorn model to be trained
        @param target_image the target image to train on, either as a numpy
                            array or a torch tensor.
        @param learning_rate initial learning rate for the Adam optimizer
                             (overridden if load_checkpoint is used).
        @param train_points_per_block number of training point that are drawn
                                      in each block (using the sampler). If not
                                      specified, it is set to the Acorn model's
                                      feature grid size.
        @param sampler an optional sampler object can be passed to sample
                                   training coordinates from each block,
                                   otherwise a uniform random sampler is used.
        @param eval_on_cpu run the ground truth evaluation on cpu
                           (for resample_training_points()) even if the model
                           is on GPU. This makes the evaluation slower but is
                           the only possibility when the image exceeds the GPU
                           memory.
        """
        self.acorn = acorn
        self.optimizer = optim.Adam(acorn.parameters(), lr=learning_rate)
        self.sampler = sampler if sampler is not None else PixelAlignedSampler()

        self.eval_on_cpu = eval_on_cpu

        self.epoch = 0
        self.loss_log = []

        self.per_block_local_coords = None
        self.per_block_targets = None

        if train_points_per_block is None:
            self.train_points_per_block = np.prod(acorn.feature_grid_resolution[:-1])
        else:
            self.train_points_per_block = train_points_per_block

        if type(target_image) == torch.Tensor:
            self.target_image_torch = target_image
        else:
            self.target_image_torch = torch.from_numpy(target_image).float().to(acorn.device)

        self.acorn.minimum_block_size = minimum_block_pixel_size / min(self.target_image_torch.size()[:2])

    def train(self,
        epoch_count=500,
        preview_output_period=-1,
        plot_period=-1,
        callbacks=()):
        """
        Main training loop. All iterations use the same random input points

        @param epoch_count number of iterations of the main training loop
        @param plot_period update plot every this number of epochs
        @param callbacks list of TrainingCallback to be called in the training
                         loop, with the trainer as first argument.
        """
        acorn = self.acorn

        if self.per_block_local_coords is None:
            self.resample_training_points()

        if plot_period > 0:
            self.figure = TrainingFigure(self, preview_output_period)

        for i in range(epoch_count):
            timer = Timer()
            self.optimizer.zero_grad()

            subtimer = Timer()
            per_block_outputs = acorn.evaluate_model_per_block(self.per_block_local_coords)
            profiling_counters['evaluate_model_per_block'].add_sample(subtimer)

            subtimer = Timer()
            loss = acorn.measure_loss(self.per_block_targets, per_block_outputs)
            profiling_counters['measure_loss'].add_sample(subtimer)

            subtimer = Timer()
            loss.backward()
            profiling_counters['backward'].add_sample(subtimer)

            self.loss_log.append(float(loss.detach()))
            self.optimizer.step()
            profiling_counters['epoch'].add_sample(timer)

            if period_starts(self.epoch, plot_period):
                subtimer = Timer()
                self.per_block_outputs = per_block_outputs
                self.figure.plot(self)
                profiling_counters['plot'].add_sample(subtimer)

            # Run extra callbacks, if appropriate
            for cb in callbacks:
                cb.maybe_call(self.epoch, self)

            self.epoch += 1

            profiling_counters['training loop'].add_sample(timer)

    def optimize_partition(self):
        """
        Wraps a call to optimize_partition on the Acorn model.

        This method can be used as in a TrainingCallback passed in the
        `extra_callbacks` argument of self.train().
        """
        print("Optimizing partition...")
        timer = Timer()
        self.acorn.optimize_partition()
        profiling_counters['optimize_partition'].add_sample(timer)
        self.resample_training_points()

    @torch.no_grad()
    def resample_training_points(self):
        """
        The training procedure uses a random set of points from the input
        domain as training input. This set uniform across active blocks, which
        means that areas of the input domain with more (smaller) active blocks
        gets more training data (because they are more detailed areas).

        NB: This is automatically called by self.train() if it has not been
        called manually, so in a normal scenario you (as a user of the Trainer
        class) do not have to pay attention to this method.

        This method can be used as in a TrainingCallback passed in the
        `extra_callbacks` argument of self.train().
        """
        timer = Timer()

        acorn = self.acorn
        device = acorn.device
        active_blocks = acorn.tree.get_active_blocks()
        dimension = acorn.dimension

        subtimer = Timer()
        self.per_block_trainset = [
            self.sampler.sample_from_block(block, self.train_points_per_block, self.target_image_torch)
            for block in active_blocks
        ]

        self.per_block_local_coords = [ local_coords for local_coords, _ in self.per_block_trainset ]
        self.per_block_targets = [ targets for _, targets in self.per_block_trainset ]
        profiling_counters['resample_training_points:sampling'].add_sample(subtimer)

        must_evaluate_targets = self.per_block_targets[0] is None
        if must_evaluate_targets:
            # Also update the ground truth for the sampled input coordinates
            subtimer = Timer()
            if self.eval_on_cpu:
                device = self.target_image_torch.device
                per_block_local_coords_cpu = [ x.cpu() for x in self.per_block_local_coords ]
                per_block_targets_cpu = acorn.evaluate_target_per_block(self.target_image_torch.cpu(), per_block_local_coords_cpu)
                self.per_block_targets = [ x.to(device) for x in per_block_targets_cpu ]
            else:
                self.per_block_targets = acorn.evaluate_target_per_block(self.target_image_torch, self.per_block_local_coords)
            profiling_counters['resample_training_points:evaluate_target_per_block'].add_sample(subtimer)

        profiling_counters['resample_training_points'].add_sample(timer)

    def save_checkpoint(self, checkpoint_filename, **extra_data):
        """
        Save the current state of the model and its training setup in a file
        that can later be loaded using self.load_checkpoint()

        @param checkpoint_filename name of the file (aboslute or relative to
                                   the current working directory) where to save
                                   the training state.
        Extra keyword arguments can be passed to be saved as is alongside the
        model data.
        """
        ensure_parent_dir(checkpoint_filename)

        torch.save({
            'model_state_dict': self.acorn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'loss_log': self.loss_log,
            'checkpoint_version': 1,
            **extra_data,
        }, checkpoint_filename)

    def load_checkpoint(self, checkpoint_filename):
        """
        Load a training set previously saved using self.save_checkpoint().

        @param checkpoint_filename name of the file (aboslute or relative to
                                   the current working directory) from which
                                   the training state gets loaded.
        """
        checkpoint = torch.load(checkpoint_filename)
        checkpoint_version = checkpoint.get('checkpoint_version', 1)
        if checkpoint_version == 1:
            self.acorn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss_log = checkpoint['loss_log']
        else:
            print(f"Unknown checkpoint version: {checkpoint_version}")
            raise NotImplemented
