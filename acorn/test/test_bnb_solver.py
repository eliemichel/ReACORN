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

from .. import bnb_solver
from ..numpy_utils import vectorized_dot

def run_all():
	np.random.seed(0)
	w = np.random.random((24, 3))
	dimension = 2
	sibling_count = 2 ** dimension

	def test(max_block):
		best_solution, best_energy = bnb_solver.solve(w, max_block, dimension)
		print(f"best_solution = {best_solution}")
		print(f"best_energy = {best_energy}")
		block_partition = np.eye(3)[best_solution]
		block_count = sum(block_partition @ np.array([1 / sibling_count, 1, sibling_count]))
		print(f"block_count = {block_count}")
		assert(block_count <= max_block)

	test(20)
	test(32)
