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

def index_select_multidim_nojit(input, indices):
    """
    Like torch.index_select, but on multiple dimensions,
    on first dimensions only (0, 1, ...)
    NB: Clamp indices if out of range

    @param input tensor of size (n_1, .., n_k, ...u)
    @param indices tensor of integers of size (...v, k)
    @return tensor of size (...v, ...u)
    """
    n = indices.size()[-1]
    input_size = input.size()

    tpl = tuple(
        indices[...,i].clamp(0, s-1)
        for i, s in enumerate(input_size[:n])
    )

    return input[(*tpl, ...)]

@torch.jit.script
def index_select_multidim_jit(input, indices):
    """
    Manual unroll of index_select_multidim_nojit
    """
    n = indices.size()[-1]
    input_size = input.size()

    if n == 1:
        return input[(
            indices[...,0].clamp(0, input_size[0]-1),
        ...)]
    elif n == 2:
        return input[(
            indices[...,0].clamp(0, input_size[0]-1),
            indices[...,1].clamp(0, input_size[1]-1),
        ...)]
    elif n == 3:
        return input[(
            indices[...,0].clamp(0, input_size[0]-1),
            indices[...,1].clamp(0, input_size[1]-1),
            indices[...,2].clamp(0, input_size[2]-1),
        ...)]
    else:
        raise NotImplemented

index_select_multidim = index_select_multidim_jit
