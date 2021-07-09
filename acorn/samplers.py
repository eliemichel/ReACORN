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

import torch

from .domain_tree import DomainTree
from .torch_utils import index_select_multidim
from .profiling import Timer, profiling_counters

# -----------------------------------------------------------------------------

class Sampler:
    """
    A sampler is called each time the DomainTree is rebalanced to allocate
    training points to each active block.
    """
    def sample_from_block(self, block, sample_count, input_image):
        """
        Sampling strategies depend on subclasses. It can be random, or grid based.
        If a sampler returns None as targets, it always returns None, it cannot
        depend on the call arguments.

        @return (
            local_coords a (sample_count,dimension) tensor of input points in
                         [-1,1].
            targets None, or a (sample_count,signal_dimension) tensor of
                    expected output. If None, it is up to the caller to use
                    acorn.evaluate_target_per_block().
        )
        """
        raise NotImplemented

# -----------------------------------------------------------------------------

class RandomSampler(Sampler):
    def sample_from_block(self, block, sample_count, input_image):
        device = input_image.device
        dimension = block.index.dimension
        local_coords = torch.rand((sample_count, dimension), device=device) * 2 - 1
        #domain_coords = DomainTree.to_input_coordinate(block.index, local_coords)
        return local_coords, None

# -----------------------------------------------------------------------------

class PixelAlignedSampler(Sampler):
    """
    A sampler that align samples to the center of pixels, so that evaluating
    target does not require linear interpolation and is thus faster
    """

    def sample_from_block(self, block, sample_count, input_image):
        device = input_image.device
        dimension = block.index.dimension
        resolution = torch.tensor([input_image.size()[:-1]], device=device)
        center = torch.from_numpy(block.index.center).float().to(device)
        radius = torch.from_numpy(block.index.radius).float().to(device)

        local_coords = torch.rand((sample_count, dimension), device=device) * 2 - 1
        domain_coords = center + local_coords * radius

        # Round in input domain
        pixel_coords = torch.floor((domain_coords * 0.5 + 0.5) * (resolution - 1))

        # Evaluate targets by simple selection
        targets = index_select_multidim(input_image, pixel_coords.long())

        # Convert back to local coords
        domain_coords = (pixel_coords / (resolution - 1)) * 2 - 1
        local_coords = (domain_coords - center) / radius

        return local_coords, targets

# -----------------------------------------------------------------------------
