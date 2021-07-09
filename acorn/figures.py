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
from matplotlib import pyplot as plt

from .domain_tree import DomainTree
from .utils import period_starts

# -----------------------------------------------------------------------------

class TrainingFigure:
    """
    Plot displayed while training an ACORN model
    """

    def __init__(self, trainer, preview_output_period=-1):
        acorn = trainer.acorn
        target_image_torch = trainer.target_image_torch

        target_image = target_image_torch.detach().cpu().numpy()

        output_image = acorn.evaluate_image((256,256), draw_quadtree=True)
        self.output_image = np.clip(output_image, 0, 1)

        self.preview_output_period = preview_output_period

        plt.ion()
        self.fig, self.axs = plt.subplots(2,2)
        self.axs[0,0].imshow(self.output_image)
        self.axs[1,0].set_yscale('log')
        self.axs[0,1].imshow(target_image)
        plt.show()

    def plot(self, trainer):
        acorn = trainer.acorn
        epoch = trainer.epoch
        loss_log = trainer.loss_log
        per_block_local_coords = trainer.per_block_local_coords
        per_block_outputs = trainer.per_block_outputs

        self.axs[0,0].clear()
        self.axs[1,1].clear()

        if period_starts(epoch, self.preview_output_period):
            output_image = acorn.evaluate_image((256,256), draw_quadtree=True)
            self.output_image = np.clip(output_image, 0, 1)

        self.axs[0,0].imshow(self.output_image)
        self.axs[1,0].plot(loss_log, color='b')

        for block, local_coords, targets_torch in zip(acorn.tree.get_active_blocks(), per_block_local_coords, per_block_outputs):
            if type(local_coords) == torch.Tensor:
                local_coords_np = local_coords.cpu().numpy()
            else:
                local_coords_np = local_coords
            xys = DomainTree.to_input_coordinate(block.index, local_coords_np) * np.array((256,256))
            targets = targets_torch.detach().cpu().numpy()
            self.axs[1,1].scatter(xys[:,0], xys[:,1], s=2, c=targets.clip(0,1))

        plt.draw()
        plt.pause(0.001)
        