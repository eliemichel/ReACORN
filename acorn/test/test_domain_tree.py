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
from matplotlib import pyplot as plt

from ..utils import plot_tree
from ..domain_tree import DomainTree

def run_all():
    tree = DomainTree(dimension=2)
    tree.root.split()
    tree.root.children[0*1 + 1*2].split()

    plot_tree(tree)

    # Test coord convertion
    x = np.array([-0.1, 0.2])
    plt.scatter(x[0], x[1])

    x_g, x_l = tree.split_input_coordinate(x)
    print(f"x_g = {x_g.raw}")
    print(f"x_l = {x_l}")

    xp = DomainTree.to_input_coordinate(x_g, x_l)
    print(f"xp = {xp}")
    assert(np.isclose(xp, x).all())

    # Test vectorized coord convertion
    x = np.array([
        [-0.1, 0.4],
        [0.6, 0.3],
        [-0.7, -0.6],
    ])
    plt.scatter(x[:,0], x[:,1])

    x_g, x_l = tree.split_input_coordinate(x)
    print(f"x_g = {x_g.raw}")
    print(f"x_l = {x_l}")

    xp = DomainTree.to_input_coordinate(x_g, x_l)
    print(f"xp = {xp}")
    assert(np.isclose(xp, x).all())

    #plt.show()
