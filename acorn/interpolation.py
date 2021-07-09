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

from .torch_utils import index_select_multidim

# -----------------------------------------------------------------------------

def linearInterpolation(feature_grid, local_coords, method="grid_sample"):
    """
    Multilinear interpolation of the (N_1, ..., N_d_in, C) feature grid
    according to d_in-vectors in [-1,1].
    Called LinInterp in the paper. This implementation is vectorized,
    meaning that the argument can be arrays of arguments.

    NB: This functions relies on torch.grid_sample(), which only works for
    dimensions 2 and 3.

    @param feature_grid (batch_size, N_1, ..., N_d_in, C) array of feature
                        grids (batch_size dim is optional).
    @param local_coords (batch_size, d_in) array of local coordinates in [-1,1]
    @param method there are two implementations of this function, one based on
                  torch's "grid_sample" method (default), the other one called
                  "simple", that is hand crafted. Simple is around 2 to 5 times
                  slower (both at forward time and at backward time) but
                  requires much less GPU memory. So we use "simple" for target
                  evaluation and "grid_sample" in the trained model.
    @return (batch_size, C) array of interpolated features
    """
    if method == "grid_sample":
        return linearInterpolation_grid_sample(feature_grid, local_coords)
    elif method == "simple":
        return linearInterpolation_simple(feature_grid, local_coords)
    else:
        raise ValueError(f"Unknown method: {method} (possible values are 'grid_sample' and 'simple')")

# -----------------------------------------------------------------------------

def linearInterpolation_grid_sample(feature_grid, local_coords):
    """
    Recommended implementation of linearInterpolation, based on
    torch's grid_sample() built-in.
    """
    device = feature_grid.device

    # This fonction mostly consists in reordering dimensions to match what
    # grid_sample expects.

    coords_shape = local_coords.size()
    coords_batched = len(coords_shape) == 2
    batch_size = coords_shape[0] if coords_batched else 1
    dimension = coords_shape[-1]

    # reshape ([batch_size,] N_1, ..., N_d_in, C)
    # into (batch_size, C, N_1, ..., N_d_in)
    grid_batched = len(feature_grid.size()) == dimension + 2
    if not grid_batched:
        # repeat batch_size times the feature grid without copying memory
        feature_grid = feature_grid.expand(batch_size, *feature_grid.size())
    dims = list(range(len(feature_grid.size())))
    feature_grid_tr = feature_grid.permute(0, dims[-1], *dims[1:-1])

    # reshape ([batch_size,] d_in)
    # into (batch_size, 1, 1, d_in)
    if not coords_batched:
        local_coords = local_coords.unsqueeze(0)
    grid_coords = local_coords
    for _ in range(dimension):
        grid_coords = grid_coords.unsqueeze(1)

    # switch coords from x, y, z to z, y, x to match grid_sample behavior
    grid_coords = torch.flip(grid_coords, (-1,))

    # Core interpolation
    interpolated = F.grid_sample(feature_grid_tr, grid_coords, align_corners=True)

    # N, C, 1, 1
    for _ in range(dimension):
        interpolated = interpolated.squeeze(-1)
    if not coords_batched:
        interpolated = interpolated.squeeze(0)

    return interpolated

# -----------------------------------------------------------------------------

@torch.jit.script
def normalized_to_scaled(normalized_coords, resolution):
    """
    @param normalized_coords batch of coordinates in range [-1,1]
    @return the same batch with coordinates in [0,resolution-1]
    """
    return (normalized_coords * 0.5 + 0.5) * (resolution - 1)

@torch.jit.script
def bilerp(values, t):
    return torch.lerp(
        torch.lerp(values[0b00], values[0b01], t[:,0:1]),
        torch.lerp(values[0b10], values[0b11], t[:,0:1]),
        t[:,1:2],
    )

@torch.jit.script
def trilerp(values, t):
    return torch.lerp(
        bilerp(values[:0b100], t),
        bilerp(values[0b100:], t),
        t[:,2:3],
    )

def linearInterpolation_simple(feature_grid, local_coords):
    """
    Reimplementation of linearInterpolation() without using torch.grid_sample
    Slower, but may use less memory because it does not allow feature_grid to
    be a batch of grids.
    """
    device = feature_grid.device
    feature_grid_size = feature_grid.size()
    grid_resolution = torch.tensor(feature_grid_size[:-1]).to(device)
    feature_size = feature_grid_size[-1]
    dimension = len(grid_resolution)

    local_coords_scaled = normalized_to_scaled(local_coords, grid_resolution)
    local_coords_floor = local_coords_scaled.floor().long()
    local_coords_fract = local_coords_scaled - local_coords_floor

    bs = torch.empty((2 ** dimension, 1, dimension), device=device, dtype=torch.long)
    for i in range(2 ** dimension):
        for c in range(dimension):
            bs[i,0,c] = i >> c & 1

    local_coords_corners = local_coords_floor.repeat(2 ** dimension, 1, 1)
    local_coords_corners += bs

    feature_grid_at_corners = index_select_multidim(feature_grid, local_coords_corners)

    if dimension == 2:
        interpolated_features = bilerp(feature_grid_at_corners, local_coords_fract)

    elif dimension == 3:
        interpolated_features = trilerp(feature_grid_at_corners, local_coords_fract)

    return interpolated_features

# -----------------------------------------------------------------------------
