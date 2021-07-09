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
import torch.nn.functional as F

from ..interpolation import linearInterpolation
from ..profiling import Timer, profiling_counters
from ..torch_utils import index_select_multidim

# -----------------------------------------------------------------------------

def run(
    grid_resolution = 512,
    feature_size = 3,
    batch_size = 256,
    dimension = 3,
    ):
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Check consistency of implementations
    for i in range(10):
        feature_grid = torch.rand([grid_resolution] * dimension + [feature_size])
        local_coords = torch.rand((batch_size, dimension)) * 2 - 1

        interpolated_features = linearInterpolation(feature_grid, local_coords, method="grid_sample")
        interpolated_features_reference = linearInterpolation(feature_grid, local_coords, method="grid_sample")
        errors = ~torch.isclose(interpolated_features, interpolated_features_reference)
        if errors.any():
            print(f"max error: {(interpolated_features - interpolated_features_reference).max()}")
        assert(not errors.any())

    # Benchmark
    profiling_counters['linearInterpolation'].reset()
    for i in range(50):
        feature_grid = torch.rand([grid_resolution] * dimension + [feature_size])
        local_coords = torch.rand((batch_size, dimension)) * 2 - 1

        timer = Timer()
        interpolated_features = linearInterpolation(feature_grid, local_coords, method="grid_sample")
        profiling_counters['linearInterpolation'].add_sample(timer)

    profiling_counters['linearInterpolation_reference'].reset()
    for i in range(50):
        feature_grid = torch.rand([grid_resolution] * dimension + [feature_size])
        local_coords = torch.rand((batch_size, dimension)) * 2 - 1

        timer = Timer()
        interpolated_features_reference = linearInterpolation(feature_grid, local_coords, method="simple")
        profiling_counters['linearInterpolation_reference'].add_sample(timer)

    # Check that it matches when using pixel aligned local coords
    pixel_coords = torch.randint(grid_resolution, (batch_size, dimension))
    pixel_coords[0,0] = 0
    pixel_coords[0,1] = 1

    local_coords = pixel_coords.float() / (grid_resolution - 1) * 2 - 1

    profiling_counters['multirow indexing:naive'].reset()
    for i in range(10):
        timer = Timer()
        ground_truth = torch.stack([ feature_grid[tuple(c)] for c in pixel_coords ])
        profiling_counters['multirow indexing:naive'].add_sample(timer)

    profiling_counters['multirow indexing:advanced'].reset()
    for i in range(10):
        timer = Timer()
        ground_truth2 = index_select_multidim(feature_grid, pixel_coords)
        profiling_counters['multirow indexing:advanced'].add_sample(timer)
    
    assert(torch.isclose(ground_truth, ground_truth2).all())

    interpolated_features = linearInterpolation(feature_grid, local_coords, method="grid_sample")
    assert(interpolated_features.size() == (batch_size, feature_size))

    errors = ~torch.isclose(interpolated_features, ground_truth, atol=1e-5).all(axis=1)
    assert(not errors.any())

    print("Profiling:")
    print('\n'.join(profiling_counters.summary()))

# -----------------------------------------------------------------------------

def run_all():
    run(
        grid_resolution = 10,
        feature_size = 1,
        batch_size = 4,
        dimension = 2,
    )

    run(
        grid_resolution = 512,
        feature_size = 3,
        batch_size = 256,
        dimension = 2,
    )

    run(
        grid_resolution = 128,
        feature_size = 3,
        batch_size = 64,
        dimension = 3,
    )
