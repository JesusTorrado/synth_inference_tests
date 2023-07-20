"""
How to define pdf classes:


Dimensionality:
  - if dimensionality can be taken as arg, __init__ should have `dim` arg, and pass it
    when calling the parent class (PDF) `__init__()` (see below). It can have a class
    attribute called `dim_min` to define the minimum dimensionality for which the PDF can
    be defined (it will be tested automatically).
  - if fixed, it must have a class attribute ``dim`` assigned to the dimensionality.

Custom methods:
- `__init__()` needs to call the parent's method.
- `logp` that returns the log probability. Tests will use `logpdf`, which is a wrap of
  this one with timing and evaluation counter.
- [Optional] `samples(n=None)` returns a table of `n` samples. If no `n` is passed, it
  should default to a reasonable number of samples to represent the distribution.

"""

import inspect
import time
import numpy as np

from .mpi import is_main_process


class PDF():

    def __init__(self, dim=None):
        if dim is not None:
            if not self.can_be_dim(dim):
                raise ValueError(
                    f"Cannot be defined for dimension smaller than {self.dim_min}.")
            self.dim = dim
        self.n = 0
        self.t = 0
        self.bounds = None

    @property
    def NameDim(self):
        return self.__class__.__name__ + str(self.dim)

    @classmethod
    def can_be_dim(cls, dim):
        dim_as_attr = getattr(cls, "dim", None)
        if dim_as_attr is not None:
            return dim_as_attr == dim
        __init__has_dim_arg = "dim" in set(inspect.signature(cls.__init__).parameters)
        if __init__has_dim_arg:
            dim_min = getattr(cls, "dim_min", 0)
            return dim >= dim_min
        return False

    def logpdf(self, params):
        self.n += 1
        start = time.time()
        logp = self.logp(params)
        self.t += time.time() - start
        return logp

    @property
    def logprior_density(self):
        if self.bounds is not None:
            return -np.sum(np.log(self.bounds.T[1] - self.bounds.T[0]))

    def logpost(self, params):
        return self.logpdf(params) + self.logprior_density

    def samples(self, n=None):
        return None

    @property
    def logZ(self):
        return None

    def triangle_plot(self, n=None, filename=None):
        """
        Does a triangle plot with the given number of samples. Optionally exports it to
        `filename``.
        """
        if not is_main_process:
            return
        try:
            from getdist.mcsamples import MCSamples
            from getdist import plots as gdplt
        except ImportError:
            raise ImportError("Triangle plots require getdist (installable with `pip`).")
        import matplotlib.pyplot as plt
        sample = self.samples(n)
        if sample is None:
            raise NotImplementedError("Reference samples not implemented for this pdf.")
        kwargs = {"names": [f"x_{i + 1}" for i in range(self.dim)]}
        if getattr(self, "bounds", None) is not None:
            kwargs["ranges"] = {p: self.bounds[i] for i, p in enumerate(kwargs["names"])}
        gdsample = MCSamples(samples=sample, **kwargs)
        g = gdplt.get_subplot_plotter()
        g.triangle_plot(gdsample)
        if filename:
            g.export(filename)
        else:
            plt.show()
