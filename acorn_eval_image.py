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
from datetime import datetime
from matplotlib.image import imsave
import numpy as np
import torch

from acorn.models import AcornModel
from acorn.profiling import profiling_counters

parser = ArgumentParser(description=
'''Evaluate an acorn model into an image''')

# Basic arguments
parser.add_argument('checkpoint', type=str, help='filename of the model checkpoint to use')
parser.add_argument('image', type=str, help='filename where to output the image (in a format supported by matplotlib.imsave)')
parser.add_argument('resolution', type=str, help='pixel size of the output image, formatted as WidthxHeight, e.g. 128x128')

parser.add_argument('--draw_quadtree', dest='draw_quadtree', action='store_true', help='display the domain tree on top of the output image')
parser.set_defaults(draw_quadtree=False)

parser.add_argument('--draw_quadtree_only', dest='draw_quadtree_only', action='store_true', help='only draw the domain tree, onto a white image, without running the model')
parser.set_defaults(draw_quadtree_only=False)

parser.add_argument('--device', type=str, default='cpu', help="device to use, either 'cuda' or 'cpu'")

@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    acorn = AcornModel(device, dimension=2, signal_dimension=3, feature_grid_resolution=(32, 32, 16))

    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint)
    acorn.load_state_dict(checkpoint['model_state_dict'], tree_only=args.draw_quadtree_only)

    try:
        resolution = parse_resolution(args.resolution)
    except ValueError:
        print("Invalid format for resolution! Must be for instance 128x128")
        exit(1)
    width, height = resolution

    if args.draw_quadtree_only:
        image = np.ones((*resolution, 3))
        acorn.draw_quadtree(image)
    else:
        print(f"Running model...")
        image = acorn.evaluate_image(resolution, draw_quadtree=args.draw_quadtree)

    print(f"Writing model output to {args.image}...")
    imsave(args.image, np.clip(image, 0, 1))

    print("Profiling:")
    print("\n".join(profiling_counters.summary()))

def parse_resolution(resolution_string):
    """
    Parse and raise ValueError in case of wrong format
    @param resolution_string string representing a resolution, like "128x128"
    @return resolution as a tuple of integers
    """
    tokens = resolution_string.split('x')
    if len(tokens) != 2:
        raise ValueError
    return tuple(int(t) for t in tokens)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
