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
import pulp

# -----------------------------------------------------------------------------

def solve(w, limit_block_count, dimension, verbose=False):
        """
        Branch and Bound solver for reorganizing active blocks.

        @param w (n,3) array of estimated fitting error if merging, leaving as
                 is or splitting each block. Weights must all be non negative.
        @param limit_block_count maximum number of blocks (N_B in the paper).
        @param dimension of the input domain.
        @return
            best_solution a vector valued with 0, 1 or 2 telling whether blocks
                          must be respectively merged, untouched or split.
            best_energy the energy of best_solution
        """

        SolverCls = getattr(pulp, pulp.listSolvers(onlyAvailable=True)[0])
        solver = SolverCls(msg=verbose)
        
        sibling_count = 2 ** dimension
        inv_sibling_count = 1 / sibling_count

        blocks = range(len(w))
        actions = range(3)  # 0, 1, 2 for resp. merge, remain and split
        prob = pulp.LpProblem("reorganize_domain_tree", pulp.LpMinimize)

        # Binary variable to optimize
        I = pulp.LpVariable.dicts("I", (blocks, actions), cat="Binary")

        # eq. (6)
        for b in blocks:
            prob += pulp.lpSum([I[b][a] for a in actions]) == 1

        # eq. (7)
        prob += pulp.lpSum([
            I[b][0] * inv_sibling_count + I[b][1] + I[b][2] * sibling_count
            for b in blocks
        ]) <= limit_block_count

        # eq. (8) - energy to minimize
        prob += pulp.lpSum([
            pulp.lpSum([ w[b,a] * I[b][a] for a in actions ])
            for b in blocks
        ])

        # extra constraint, to ensure that neighbor blocks
        # prob += (TODO)

        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if verbose:
            print(f"Optimization solver status: {status}")

        solution = np.array([
            np.argmax([ pulp.value(I[n][a]) for a in actions ])
            for n in blocks
        ])

        return solution, pulp.value(prob.objective)

# -----------------------------------------------------------------------------
