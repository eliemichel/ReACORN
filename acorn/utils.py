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

import os
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------

def plot_tree(tree):
    def draw_lines(block):
        x, y = block.index.center
        r = block.index.radius
        plt.plot([x-r, x-r, x+r, x+r, x-r], [y-r, y+r, y+r, y-r, y-r])
    tree.visit(draw_lines)

# -----------------------------------------------------------------------------

def period_starts(counter, period):
    """
    Utility function to test whether we are at the beginning of a cycle
    of length `period`.
    If the period is non positive, this always return False, otherwise
    it returns True when `counter` is 0, cycle, 2*cycle, etc.

    @param counter typically an epoch index, or a frame
    @param period the number of epochs between two positive outputs of this
                  function
    @return True iff the counter is at the beginning of a cycle
    """
    return period > 0 and counter % period == 0

def period_ends(counter, period):
    """
    Same as @see `period_starts()` but telling the last index of the cycle,
    ie. returns True when `counter` is cycle-1, 2*cycle-1, etc.
    """
    return period > 0 and counter % period == period - 1

# -----------------------------------------------------------------------------

def ensure_parent_dir(filepath):
    """
    Make sure that the parent directory of a filepath exists
    """
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)
