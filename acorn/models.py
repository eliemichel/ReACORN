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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi, sqrt

from .domain_tree import DomainTree
from .interpolation import linearInterpolation
from . import bnb_solver
from .profiling import Timer, profiling_counters

# -----------------------------------------------------------------------------

class CoordinateEncoderNet(nn.Module):
    """
    Model that translates an input global grid coordinate into a grid of
    features, to later be interpolated using local coordinates.

    The model does positional encoding of the input coordinate using random
    Fourier features, then has a few hidden layers.

    The model outputs a feature grid shaped as a vector, it is up to the caller
    code to reshape it to (N_1, ..., N_d_in, C)
    """
    def __init__(self,
            domain_dimension,
            feature_grid_size,
            fourier_feature_size=16,
            fourier_feature_std=1.0,
            hidden_layer_size=512,
            hidden_layer_count=4):
        """
        @param domain_dimension number of dimensions of the acorn input domain
                                (typically 2 or 3)
        @param feature_grid_size total number of features in the output feature
                                 grid, i.e. N_1 * ... * N_d_in * C
        @param fourier_feature_size number of random Fourier feature weights
        @param fourier_feature_std standard deviation of the Fourier feature
                                   weights
        @param hidden_layer_size number of neurons per hidden layer
        @param hidden_layer_count number of hidden layers (at least 1)
        """
        super().__init__()
        # Random weights for Fourier features
        B = 2 * pi * torch.normal(0, fourier_feature_std, size=(domain_dimension + 1, fourier_feature_size))
        self.register_buffer("B", B)  # make sure this tensor is moved to the same device as the model
        self.ff_normalizer = 1 / sqrt(fourier_feature_size)

        seq = [
            nn.Linear(2 * fourier_feature_size, hidden_layer_size),
            nn.ReLU(),
        ]

        for _ in range(hidden_layer_count - 1):
            seq.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            seq.append(nn.ReLU())

        seq.append(nn.Linear(hidden_layer_size, feature_grid_size))

        self.mlp = nn.Sequential(*seq)
    
    def forward(self, x):
        Bx = x @ self.B
        x = self.ff_normalizer * torch.cat((torch.cos(Bx), torch.sin(Bx)), axis=1)
        x = self.mlp(x)
        return x

# -----------------------------------------------------------------------------

class DecoderNet(nn.Module):
    """
    The decoder takes an interpolated feature vector and turn it into the
    output signal. This net is intended to be very lightweight, it has only one
    hidden layer.
    """
    def __init__(self, feature_size, signal_dimension, hidden_layer_size=64):
        """
        @param feature_size dimension of an input feature vector (C)
        @param signal_dimension number of component in the output signal
                                e.g. 3 or 4 for an image, 1 for a signed
                                distance field, etc.
        @param hidden_layer_size number of neurons in the hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(feature_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, signal_dimension)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x * 0.5 + 0.5

# -----------------------------------------------------------------------------

class AcornModel:
    def __init__(self,
            device,
            dimension=2,
            signal_dimension=3,
            feature_grid_resolution=(32, 32, 16),
            max_active_blocks=512,
            alpha=0.1,
            beta=0.01,
            decoder_hidden_layer_size=64,
            fourier_feature_size=16,
            fourier_feature_std=1.0,
            encoder_hidden_layer_size=512,
            encoder_hidden_layer_count=4,
            ):
        """
        @param device torch device to run the model on
        @param dimension in the paper d_in
        @param signal_dimension dimension of the output signal (3 for an RGB
                                image, 1 for a signed distance field, etc.)
        @param feature_grid_resolution in the paper (N_0, ..., N_d_in, C)
        @param max_active_blocks in the paper N_B
        @param alpha captures the assumption that the error when merging blocks
                     is worse than 𝑁_𝑆 times the current block error
        @param beta assumes that the error when splitting a block is likely to
                    be better than 1/𝑁_S times the current error
        """
        self.device = device
        self.dimension = dimension
        self.signal_dimension = signal_dimension
        self.feature_grid_resolution = feature_grid_resolution
        self.max_active_blocks = max_active_blocks
        self.alpha = alpha
        self.beta = beta
        self.minimum_block_size = 0  # set when training

        self.tree = DomainTree(dimension=dimension)
        self.tree.root.split()

        self.encoder = CoordinateEncoderNet(dimension, np.prod(feature_grid_resolution)).to(device)
        self.decoder = DecoderNet(feature_grid_resolution[-1], signal_dimension, decoder_hidden_layer_size).to(device)

        self._testing = False

    def evaluate_model_per_block(self, per_block_local_coords, is_training=True):
        """
        Evaluate the model, with input data point listed by active block,
        in local coordinates.

        @param per_block_local_coords list of (N,dimension) torch tensors with
                                      values in [-1,1].
        @param is_training set to True iff at training time (when not training
                           we use an alternative interpolation implementation
                           that requires less memory)
        @return list of (N,signal_output) torch tensors
        """
        assert(type(per_block_local_coords[0]))
        device = self.device
        active_blocks = self.tree.get_active_blocks()
        block_count = len(active_blocks)

        # A. Evaluate feature grid for each active block
        # (perf bottleneck: self.encoder())

        # (in the paper, list of x_g^i vectors)
        all_block_indices = torch.stack([
            block.index.raw_torch(device)
            for block in active_blocks
        ])

        # (in the paper, list Gamma_i vectors)
        timer = Timer()
        all_feature_grids = self.encoder(all_block_indices)
        profiling_counters["evaluate_model_per_block:encoder"].add_sample(timer)

        all_feature_grids = all_feature_grids.reshape((-1, *self.feature_grid_resolution))


        # B. Evaluate output for each block
        # (perf bottleneck: linearInterpolation and self.decoder() on par)
        timer = Timer()

        per_block_outputs = []

        #interp_time = 0
        #decoder_time = 0
        for block, feature_grid, local_coords in zip(active_blocks, all_feature_grids, per_block_local_coords):
            # (in the paper, list of x_l^i,j vectors)
            #subtimer = Timer()
            features = linearInterpolation(feature_grid, local_coords, method='grid_sample' if is_training else 'simple')
            #interp_time += subtimer.ellapsed()
            
            #subtimer = Timer()
            block_output_torch = self.decoder(features)
            #decoder_time += subtimer.ellapsed()

            per_block_outputs.append(block_output_torch)

        #profiling_counters["evaluate_model_per_block:interp_time"].add_sample(interp_time)
        #profiling_counters["evaluate_model_per_block:decoder_time"].add_sample(decoder_time)
        profiling_counters["evaluate_model_per_block:B"].add_sample(timer)

        return per_block_outputs

    def evaluate_image(self, resolution, draw_quadtree=False):
        """
        Create an image of a given resolution from the model.
        @param resolution tuple of (width, height)
        @param draw_quadtree add debug line on top of the output to visualize
                             the quadtree.
        @return RGB image
        """
        width, height = resolution
        device = self.device

        timer = Timer()
        per_block_local_coords_torch = []

        active_blocks = self.tree.get_active_blocks()
        for block in active_blocks:
            block_width = block.index.radius * width
            block_height = block.index.radius * height
            block_size = np.concatenate((block_width, block_height)).astype(int)

            local_pixels = np.transpose(np.mgrid[:block_width,:block_height], axes=(1, 2, 0))
            local_coords = local_pixels / np.array(block_size) * 2 - 1
            local_coords_flat = local_coords.reshape((-1, 2))

            local_coords_flat_torch = torch.from_numpy(local_coords_flat).float().to(device)

            per_block_local_coords_torch.append(local_coords_flat_torch)
        profiling_counters["evaluate_image:split"].add_sample(timer)

        timer = Timer()
        per_block_output = self.evaluate_model_per_block(per_block_local_coords_torch, is_training=False)
        profiling_counters["evaluate_image:blocks"].add_sample(timer)

        timer = Timer()
        image = np.empty((height, width, 3))
        for block, block_output in zip(active_blocks, per_block_output):
            block_width = block.index.radius * width
            block_height = block.index.radius * height
            block_size = np.concatenate((block_width, block_height)).astype(int)
            block_center = (block.index.center * 0.5 + 0.5) * np.array(resolution)
            block_start = np.floor(block_center - block_size / 2).astype(int)

            x, y = block_start
            w, h = block_size

            if w * h == 0:
                continue

            image[x:x+w, y:y+h] = block_output.detach().cpu().numpy().reshape(w, h, -1)
        profiling_counters["evaluate_image:unsplit"].add_sample(timer)

        if draw_quadtree:
            self.draw_quadtree(image)

        return image

    def draw_quadtree(self, image):
        width, height = image.shape[:2]
        def draw_lines(block):
            x, y = block.index.center
            r = block.index.radius
            xmin = int(((x - r) * 0.5 + 0.5) * width)
            ymin = int(((y - r) * 0.5 + 0.5) * height)
            xmax = int(((x + r) * 0.5 + 0.5) * width)
            ymax = int(((y + r) * 0.5 + 0.5) * height)
            image[ymin,xmin:xmax] = 0
            image[ymax-1,xmin:xmax] = 0
            image[ymin:ymax,xmin] = 0
            image[ymin:ymax,xmax-1] = 0
        self.tree.visit(draw_lines)

    def evaluate_target_per_block(self, target_image, per_block_local_coords):
        """
        Extract target training data from ground truth image, that matches the
        random input points from per_block_local_coords
        @param target_image ground truth image, as a torch tensor
        @param per_block_local_coords list of (N,dimension) torch tensors with
                                      values in [-1,1]
        @return list of (N,signal_output) torch tensors
        """
        assert(type(per_block_local_coords[0]) == torch.Tensor)
        active_blocks = self.tree.get_active_blocks()

        per_block_targets = []

        for block, local_coords in zip(active_blocks, per_block_local_coords):
            coords = DomainTree.to_input_coordinate(block.index, local_coords)
            block_target_torch = linearInterpolation(target_image, coords, method="simple")
            per_block_targets.append(block_target_torch)

        return per_block_targets

    def measure_loss(self, per_block_targets, per_block_outputs):
        """
        Consolidate the overall loss of the model
        @param per_block_targets result of evaluate_target_per_block
        @param per_block_outputs result of evaluate_model_per_block
        @return torch loss
        """
        active_blocks = self.tree.get_active_blocks()

        loss_fn_torch = nn.MSELoss()
        loss = 0
        loss_divisor = 0

        assert(len(active_blocks) == len(per_block_targets))
        assert(len(active_blocks) == len(per_block_outputs))

        for block, block_target_torch, block_output_torch in zip(active_blocks, per_block_targets, per_block_outputs):
            block_error = loss_fn_torch(block_output_torch, block_target_torch)
            block.memoized_mean_error = block_error
            loss += block_error
            loss_divisor += 1

        return loss / loss_divisor

    def optimize_partition(self):
        """
        Re-balance the domain tree, to be called e.g. every 100 training epochs
        """
        MergeAction = 0
        RemainAction = 1
        SplitAction = 2

        sibling_count = 2 ** self.dimension

        active_blocks = self.tree.get_active_blocks()
        per_block_mean_error = np.array([
            block.memoized_mean_error.detach().cpu().numpy()
            for block in active_blocks
        ])
        per_block_volume = np.array([ block.index.volume for block in active_blocks ])

        # Vector "w" in the paper
        # for each block, estimate the fitting error if (a) the block is merged
        # (b) the block remains as is and (c) the block is split.
        w = np.zeros((len(active_blocks),3))

        # eq. (10)
        w[:,RemainAction] = per_block_mean_error * per_block_volume

        # eq. (11)
        for i, block in enumerate(active_blocks):
            parent_err = block.parent.memoized_mean_error
            if parent_err is not None:
                w[i,MergeAction] = (1 / sibling_count) * parent_err * block.parent.index.volume
            else:
                w[i,MergeAction] = (sibling_count + self.alpha) * w[i,RemainAction]

        # eq. (12)
        for i, block in enumerate(active_blocks):
            if block.size <= self.minimum_block_size:
                w[i,SplitAction] = 0
            elif block.memoized_children_errors:
                w[i,SplitAction] = sum(block.memoized_children_errors) * block.index.volume
            else:
                w[i,SplitAction] = max(0, 1 / sibling_count - self.beta) * w[i,1]

        # Prevent from merging blocks whose all siblings are not active
        cannot_be_merged = [
            not all([sibling.is_active for sibling in block.parent.children[:]])
            for block in active_blocks
        ]
        w[cannot_be_merged,MergeAction] = w[cannot_be_merged].max(axis=1) + 1

        block_partition, energy = bnb_solver.solve(w, self.max_active_blocks, self.dimension)

        for block, action, cannot_merge in zip(active_blocks, block_partition, cannot_be_merged):
            if action == MergeAction and block.parent.can_merge:
                # (block might no longer be active if one of its sibling
                # already performed the merging operation, or one of its
                # siblings may have been split and hence this can no longer
                # be merged.)
                block.parent.merge_children()

            if action == SplitAction:
                block.split()

    # ======= Houskeeping methods ======= #

    def parameters(self):
        return [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "tree": self.tree.state_dict(),
        }

    def load_state_dict(self, state_dict, tree_only=False):
        if not tree_only:
            self.encoder.load_state_dict(state_dict['encoder'])
            self.decoder.load_state_dict(state_dict['decoder'])
        self.tree.load_state_dict(state_dict['tree'])

