"""
MPI and other parallelization utils.
"""

import os

try:
    from mpi4py import MPI
    try:
        import dill
        # Use dill pickler (can seriealize more stuff, e.g. lambdas)
        MPI.pickle.__init__(dill.dumps, dill.loads)
    except Exception:
        pass
    # Define some interfaces
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()
    is_main_process = not bool(mpi_rank)
except Exception:
    mpi_comm = None
    mpi_size = 0
    mpi_rank = 0
    is_main_process = True
multiple_processes = mpi_size > 1


def get_num_threads():
    """
    Tries to guess the number of available threads per process.
    """
    n_total_available = len(os.sched_getaffinity(0)) - mpi_size + 1
    n_OMP = os.getenv("OMP_NUM_THREADS")
    if n_OMP is not None:
        n_OMP = int(n_OMP)
        return min(n_total_available, n_OMP)
    return n_total_available
