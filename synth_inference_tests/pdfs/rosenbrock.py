"""
Generalization of the Rosenbrock test function to arbitrary dimensions from
https://arxiv.org/abs/1110.2997 (via https://arxiv.org/abs/2306.16923).

Difficulty: pronounced curving degeneracies.

"""

from warnings import warn

import numpy as np

from ..pdf import PDF


class Rosenbrock(PDF):
    """
    Generalized Rosenbrock function from https://arxiv.org/abs/1110.2997
    """

    dim = None
    dim_min = 2
    multimodal = False
    nongaussian = True

    def __init__(self, dim):
        super().__init__(dim)
        self.bounds = np.array(dim * [[-5, 5]])
        self.ref_bounds = np.array(dim * [[0.5, 1.5]])

    def logp(self, params):
        params = np.atleast_2d(params)
        return -np.sum(
            (1 - params[:, :-1]) ** 2 + 100 * (params[:, 1:] - params[:, :-1] ** 2) ** 2,
            axis=-1,
        )

    @property
    def logZ(self):
        return {
            # PolyChord (nlive=200d/num_repeats=50d/prec_crit=0.001), avg 20 realisations
            # 2: -5.824,  # +/- 0.025
            2: -5.804,  # analytic, from arXiv:1110.2997
            3: -10.47,  # +/- 0.03
            4: -15.08,  # +/- 0.03
            5: -19.67,  # +/- 0.03
            6: -24.34,  # +/- 0.03
            7: -28.92,  # +/- 0.03
            8: -33.65,  # +/- 0.03
            9: -38.37,  # +/- 0.04
            10: -43.01,  # +/- 0.04; -43.2 +/- ~0.2, from arXiv:2306.16923
        }.get(self.dim)
