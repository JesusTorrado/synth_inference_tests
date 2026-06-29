"""
Generalization of the Rosenbrock test function to arbitrary dimensions from
https://arxiv.org/abs/1110.2997 (via https://arxiv.org/abs/2306.16923).

Difficulty: pronounced curving degeneracies.
"""

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
            2: -5.804,  # analytic, from arXiv:1110.2997; Ours: -5.825 +/- 0.025
            3: -10.465,  # +/- 0.025
            4: -15.08,  # +/- 0.03
            5: -19.69,  # +/- 0.03
            6: -24.36,  # +/- 0.03
            7: -28.935,  # +/- 0.035
            8: -33.650,  # +/- 0.035
            9: -38.410,  # +/- 0.035
            10: -43.03,  # +/- 0.04; -43.2 +/- ~0.2, from arXiv:2306.16923
            12: -52.47,  # +/- 0.04
            16: -71.40,  # +/- 0.04
            20: -90.44,  # +/- 0.04
            24: -109.37,  # +/- 0.04
        }.get(self.dim)
