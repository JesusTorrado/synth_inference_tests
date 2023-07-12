"""
How to define pdf classes:


Dimensionality:
  - if dimensionality can be taken as arg, __init__ should have `dim` arg, and set it as
    an attribute called "dim".
  - if fixed, it must have a class attribute ``dim`` assigned to the dimensionality.

Custom methods:
- `__init__()` needs to call the parent's method.
- `logp` that returns the log probability. Tests will use `logpdf`, which is a wrap of
  this one with timing and evaluation counter.

"""

import inspect
import time

class PDF():

    def __init__(self):
        self.n = 0
        self.t = 0


    @classmethod
    def can_be_dim(cls, dim):
        dim_as_attr = getattr(cls, "dim", None)
        if dim_as_attr is not None:
            return dim_as_attr == dim
        __init__has_dim_arg = "dim" in set(inspect.signature(cls.__init__).parameters)
        if __init__has_dim_arg:
            return True
        return False

    def logpdf(self, *params):
        self.n += 1
        start = time.time()
        logp = self.logp(params)
        self.t += time.time() - start
        return logp
