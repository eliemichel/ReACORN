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

# -----------------------------------------------------------------------------

class BlockIndex:
    """
    A block index x_g is a vector of size dimension + 1 uniquely identifying
    a block from a DomainTree. A block is a cube centered at x_g.center and
    whose edges have size 2 * x_g.radius

    The vector itself can be queried by calling x_g.raw
    Inversely, convert any raw vector into a BlockIndex by using the default
    constructor.

    This class is vectorized, meaning that 'raw' can be either a single vector
    or an array of vectors, in which case this object represents an array
    of block indices.
    """
    def __init__(self, raw):
        self.raw = raw

    @property
    def center(self):
        """
        Coordinate of the center of the block, in the original normalized
        coordinate domain [-1,1]²
        """
        return self.raw[...,:-1]

    @property
    def radius(self):
        """
        Half size of an edge of the block. The root block has a radius of 1.0,
        then each subblock has a radius that is half the one of its parent.
        """
        return self.raw[...,-1:]

    @property
    def dimension(self):
        """
        Number of dimensions of the input domain associated to this block index
        (shape - 1 because the block index has an extra dimension to store the
        block's size.)
        """
        return self.raw.shape[-1] - 1

    @property
    def volume(self):
        """
        Volume of the block
        """
        return self.dimension * self.dimension * self.dimension

    def raw_torch(self, device):
        return torch.from_numpy(self.raw).float().to(device)

    def get_linearized_radius(self, tree):
        """
        In the original ACORN paper, the last dimension of the block index is
        the (normalized) discrete level rather than the radius. This returns
        the normalized level.
        @param tree the tree this block belongs to, to get the max_depth
                    (for normalization)
        @return value in [0,1]
        """
        return -log2(self.radius) / tree.max_depth

    def __hash__(self):
        if self.raw.ndim != 1:
            raise TypeError("Batched BlockIndex cannot be hashed, only single element ones can.")
        return hash(tuple(self.raw * 0.5 + 0.5))

    def __eq__(self, other):
        if self.raw.ndim != 1:
            raise TypeError("Batched BlockIndex cannot be compared, only single element ones can.")
        return tuple(self.raw) == tuple(other.raw)

    def __iter__(self):
        # When a BlockIndex `block_indices` represents a batch of indices, one
        # can iterate over it using `for index in block_indices`.
        if self.raw.ndim == 1:
            yield self
        else:
            for single_index in self.raw:
                yield BlockIndex(single_index)

# -----------------------------------------------------------------------------

class Block:
    """
    Octant of a DomainTree, with recursive part of DomainTree's methods.
    When a docstring is missing, see equivalently named method in DomainTree.
    NB1: A block is active iff it has no children.
    NB2: A block goes from -radius included to +radius excluded
    """
    def __init__(self, raw_block_index):
        self.index = BlockIndex(np.array(raw_block_index))
        self.children = []
        self.parent = None

        # Every time we update the error estimation for this block, we save
        # the value here. This remains even when the block is no longer active
        # because we need it to decide whether we should merge its children.
        self.memoized_mean_error = None

        # When merging children, we remove them from self.children but keep
        # their estimation error around to decide whether we should split
        # it again.
        self.memoized_children_errors = []

    @property
    def is_active(self):
        return not self.children

    @property
    def can_merge(self):
        if self.is_active:
            return False
        for child in self.children:
            if not child.is_active:
                return False
        return True

    def split(self):
        """
        Assuming this is an active block, split it into new subblocks.
        The subblocks are then active while this one no longer is.
        """
        assert(self.is_active)
        dimension = self.index.dimension
        
        for i in range(2 ** dimension):
            # child index is such that its bits tell the sign of (x-center)
            is_positive = [i >> b & 1 for b in range(dimension)]
            child_index_offset = (np.array(is_positive + [0]) - 0.5) * self.index.radius
            child = Block(self.index.raw + child_index_offset)
            child.parent = self
            self.children.append(child)

        # remove previous children that were still cached
        self.memoized_children_errors = []

    def merge_children(self):
        """
        Assuming all children are active, merge them.
        """
        assert(self.can_merge)

        self.memoized_children_errors = [
            child.memoized_mean_error
            for child in self.children
        ]

        self.children = []

    def find_active_block(self, x):
        """
        @param x is assumed to be within the bounds of the block
        """
        if self.is_active:
            if x.ndim == 1:
                return self.index
            else:
                return BlockIndex(np.tile(self.index.raw, (len(x), 1)))

        # child index is such that its bits tell the sign of (x-center)
        is_positive = (x - self.index.center) > 0
        bits = 2 ** np.arange(x.shape[-1])  # e.g. [1,2,4]
        child_index = (is_positive * bits).sum(axis=-1)
        if child_index.ndim == 0:
            assert(x.ndim == 1)
            return self.children[child_index].find_active_block(x)
        else:
            assert(len(child_index) == len(x))
            # TODO: is there a way to properly vectorize this?
            return BlockIndex(np.vstack([
                self.children[idx].find_active_block(xx).raw
                for idx, xx in zip(child_index, x)
            ]))

    def visit(self, visitor):
        if self.is_active:
            visitor(self)
        else:
            for child in self.children:
                child.visit(visitor)

    # ======= Houskeeping methods ======= #

    def state_dict(self):
        return {
            "index": list(self.index.raw),
            "children": [child.state_dict() for child in self.children],
            "memoized_mean_error": self.memoized_mean_error,
            "memoized_children_errors": self.memoized_children_errors,
        }

    def load_state_dict(self, state_dict):
        self.index = BlockIndex(np.array(state_dict['index']))

        self.children = []
        for child_state in state_dict['children']:
            child = Block(child_state["index"])
            child.load_state_dict(child_state)
            child.parent = self
            self.children.append(child)

        self.memoized_mean_error = state_dict['memoized_mean_error']
        self.memoized_children_errors = state_dict['memoized_children_errors']

# -----------------------------------------------------------------------------

class DomainTree:
    """
    Adaptive quad/oct-tree partionning the input domain
    """
    def __init__(self, dimension=2, max_depth=10):
        """
        Initialize a new domain tree with a single active block
        @param dimension number of dimensions in the input domain
                         (typically 2 or 3)
        @param max_depth maximum level index of a block. Depth index starts
                         at 0, so a max_depth of n means that there is a total
                         of n + 1 levels.
        """
        self.max_depth = max_depth
        self.root = Block([0] * dimension + [1])

        self.root.parent = self
        self.children = [self.root]
        self.memoized_mean_error = None

    def find_active_block(self, x):
        """
        @param x normalized coordinate, in domain [-1,1]², or array of such
        @return the block index of the only active block of the tree that
                contains x.
        """
        return self.root.find_active_block(x)

    def visit(self, visitor):
        """
        Visit all active nodes
        @param visitor a callback function taking the block as argument
        """
        self.root.visit(visitor)

    def get_active_blocks(self):
        """
        @return the list of active blocks in the tree
        """
        active_blocks = []
        self.visit(active_blocks.append)
        return active_blocks

    # ======= Conversion methods ======= #

    def split_input_coordinate(self, domain_coords):
        """
        Convert a global domain coordinate into
        (convert x into x_g and x_l, would say the paper)

        This method can be called either on a single domain coordinate or on
        an array of such, in which case local_coords has the same shape as
        the input and block_indices is a single BlockIndex object whose "raw"
        value is an array of coordinates.

        @param x normalized coordinate, in domain [-1,1]
        @return block indices, then coordinates within these blocks,
                in domain [-1,1] as well
        """
        block_indices = self.find_active_block(domain_coords)
        assert(len(block_indices.center) == len(domain_coords))
        local_coords = (domain_coords - block_indices.center) / block_indices.radius
        return block_indices, local_coords

    @classmethod
    def to_input_coordinate(cls, block_indices, local_coords):
        """
        Inverse of self.split_input_coordinate(). This method does not require
        to know the tree, it is a class method.

        @param block_indices instance of BlockIndex representing one or a batch
                             of global coordinates (x_g)
        @param local_coords numpy array or torch tensor of one or a batch of
                            local coordinates (x_l)
        @return (batch of) coordinates in the original input domain, as a torch
                tensor if local_coords was, or a numpy array
        """
        if type(local_coords) == torch.Tensor:
            device = local_coords.device
            center = torch.from_numpy(block_indices.center).float().to(device)
            radius = torch.from_numpy(block_indices.radius).float().to(device)
            return center + local_coords * radius
        else:
            return block_indices.center + local_coords * block_indices.radius

    # ======= Houskeeping methods ======= #

    def state_dict(self):
        return {
            "root": self.root.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.root.load_state_dict(state_dict['root'])

# -----------------------------------------------------------------------------
