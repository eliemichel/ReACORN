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

from argparse import ArgumentParser
from time import time
import numpy as np
from matplotlib.image import imread
import torch
import re

from acorn.models import AcornModel
from acorn.training import TrainingCallback, Trainer
from acorn.profiling import Timer, profiling_counters

parser = ArgumentParser(description=
'''Train an ACORN model to encode a 2D RGB image''')

# Basic arguments
parser.add_argument('image_filename', type=str, help='filename of the image to train on')
parser.add_argument('--start_checkpoint', type=str, help='filename of the checkpoint to start from')
parser.add_argument('--end_checkpoint', type=str, help='filename of the checkpoint to save state to in the end, and also during training if --save_checkpoint_period is set. This name can contain a series of # signs to contain the epoch number (padded by the number of #)')

# Model architecture (must match the original one when using --start_checkpoint)
parser.add_argument('--decoder_hidden_layer_size', type=int, default=64, help='number of neurons on the single hidden layer of the decoder')
parser.add_argument('--feature_size', type=int, default=16, help='dimension of the feature grid that the encoder outputs and the decoder uses as input')
parser.add_argument('--feature_grid_resolution', type=int, default=32, help='size per spatial dimension of the feature grid output by the encoder')
parser.add_argument('--max_active_blocks', type=int, default=1024, help='maximum number of active blocks in the domain tree')
parser.add_argument('--fourier_feature_size', type=int, default=16, help='number of random Fourier features (positional encoding) before the grid encoder')
parser.add_argument('--fourier_feature_std', type=float, default=1.0, help='standard deviation of the random Fourier feature coefficients')
parser.add_argument('--encoder_hidden_layer_size', type=int, default=512, help='number of neurons per hidden layer of the grid encoder')
parser.add_argument('--encoder_hidden_layer_count', type=int, default=4, help='number of hidden layers in the grid encoder (they all have the same size)')
parser.add_argument('--alpha', type=float, default=0.2, help='(from the paper) captures the assumption that the error when merging blocks is worse than N_S times the current block error')
parser.add_argument('--beta', type=float, default=0.02, help='(from the paper) assumes that the error when splitting a block is likely to be better than 1/N_S times the current error')

# Training
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='initial learning rate from training (Adam optimizer is used)')
parser.add_argument('--optimize_partition_period', type=int, default=-1, help='number of training epochs between two optimizations of the domain tree partition. A negative value mean to never optimize the partition.')
parser.add_argument('--resample_period', type=int, default=-1, help='number of training epochs between two resamplings of training input set in each block. A negative value mean to never resample (except when domain tree partition is re-balanced).')
parser.add_argument('--train_points_per_block', type=int, help='number of training point per active block. Defaults to feature_grid_resolution^dimension')
parser.add_argument('--eval_on_cpu', dest='eval_on_cpu', action='store_true', help='run the ground truth evaluation on cpu, even if the model is on GPU. This makes the evaluation slower but is the only possibility when the image exceeds the GPU memory')
parser.set_defaults(eval_on_cpu=False)
parser.add_argument('--minimum_block_pixel_size', type=int, default=1, help='minimal size of a block, in pixels')

# Training visualization
parser.add_argument('--preview_output_period', type=int, default=10, help='number of training epochs between two updates of the preview image. A negative value mean to never update.')
parser.add_argument('--plot_period', type=int, default=-1, help='update plot every this amount of epoch, or -1 to deactivate plotting')
parser.add_argument('--print_loss_period', type=int, default=5, help='print current loss every this amount of epoch, or -1 to never do it')
parser.add_argument('--print_profiling_period', type=int, default=50, help='print profiling measures every this amount of epoch, or -1 to never do it')
parser.add_argument('--save_checkpoint_period', type=int, default=-1, help='write the end checkpoint before the end of training, every this amount of epochs. If -1, only checkpoint at the very end.')

# System
parser.add_argument('--device', type=str, default='cpu', help="device to use, either 'cuda' or 'cpu'")

def main(args):
    device = torch.device(args.device)

    target_image = imread(args.image_filename)
    if target_image.dtype == np.uint8:
        target_image = target_image.astype(float) / 255.

    # TODO: Load model architecture from checkpoint (currently we expect the
    # user to provide the very same architecture arguments).

    print("Initializing ACORN model...")
    feature_grid_resolution = (
        args.feature_grid_resolution,
        args.feature_grid_resolution,
        args.feature_size,
    )
    acorn = AcornModel(
        device,
        dimension=2,
        signal_dimension=3,
        feature_grid_resolution=feature_grid_resolution,
        decoder_hidden_layer_size=args.decoder_hidden_layer_size,
        max_active_blocks=args.max_active_blocks,
        fourier_feature_size=args.fourier_feature_size,
        fourier_feature_std=args.fourier_feature_std,
        encoder_hidden_layer_size=args.encoder_hidden_layer_size,
        encoder_hidden_layer_count=args.encoder_hidden_layer_count,
    )

    trainer = Trainer(
        acorn,
        target_image,
        learning_rate=args.learning_rate,
        train_points_per_block=args.train_points_per_block,
        eval_on_cpu=args.eval_on_cpu,
    )

    if args.start_checkpoint:
        print(f"Loading checkpoint from {args.start_checkpoint}...")
        trainer.load_checkpoint(args.start_checkpoint)

    # Every 'period' epochs, call the function passed as first argument
    # (callback functions are defined bellow)
    training_callbacks = (
        TrainingCallback(print_loss, period=args.print_loss_period),
        TrainingCallback(print_profiling, period=args.print_profiling_period),
        TrainingCallback(save_checkpoint, period=args.save_checkpoint_period, args=args),
        TrainingCallback(Trainer.optimize_partition, period=args.optimize_partition_period),
        TrainingCallback(Trainer.resample_training_points, period=args.resample_period, at_period_start=False),
    )

    print(f"Training model for {args.epochs} epochs...")
    trainer.train(
        epoch_count=args.epochs,
        preview_output_period=args.preview_output_period,
        plot_period=args.plot_period,
        callbacks=training_callbacks,
    )

    print(f"Finished training with loss {trainer.loss_log[-1]}")

    save_checkpoint(trainer, args)

# -----------------------------------------------------------------------------
# Training callbacks

def save_checkpoint(trainer, args):
    if not args.end_checkpoint:
        return

    end_checkpoint = args.end_checkpoint
    m = re.search("#+", end_checkpoint)
    if m is not None:
        digits = m.end() - m.start()
        epoch = str(trainer.epoch).zfill(digits)
        end_checkpoint = end_checkpoint[:m.start()] + epoch + end_checkpoint[m.end():]

    print(f"Saving checkpoint to {end_checkpoint}...")
    trainer.save_checkpoint(end_checkpoint, cmd_args=vars(args), timestamp=time())

def print_loss(trainer):
    print(f"Epoch #{trainer.epoch} -- loss = {trainer.loss_log[-1]}")

def print_profiling(trainer):
    print("Profiling Counters:")
    print("\n".join(profiling_counters.summary()))

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
