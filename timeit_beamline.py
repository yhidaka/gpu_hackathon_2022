from datetime import datetime
from importlib import reload
import os
import sys
import time

# Must set these before NumPy import to disable its multithreading
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1

import cupy as cp
import h5py
import numpy as np

import pylatt as latt


def to_gpu(x):

    t0 = time.perf_counter()
    x_in_gpu = cp.asarray(x)
    print(f"CPU-to-GPU took {time.perf_counter()-t0:.3f} [s]")

    return x_in_gpu


def run():

    xmin = -1e-3
    xmax = +1e-3
    ymin = 1e-6
    ymax = 2e-3
    nturn = 256

    import nsls2sr_supercell as acell

    n_list = [10, 20, 50, 100, 200, 500, 1_000, 2_000]
    # n_list = [10, 20,50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    # n_list = [2_000, 5_000, 10_000]
    # n_list = [10, 20]
    pu_types = ["cpu", "gpu"]
    # pu_types = ['cpu']
    # pu_types = ['gpu']
    nrepeat = 3
    # nrepeat = 1

    timing = {}

    for pu_type in pu_types:

        timing[pu_type] = {}

        if pu_type == "gpu":
            latt.use_gpu()
        elif pu_type == "cpu":
            latt.use_cpu()
        else:
            raise ValueError

        reload(acell)

        for n in n_list:

            if pu_type == "cpu":
                if n > 500:
                    continue

            nx = ny = n

            print(f"Using {pu_type}: {n=}")

            dts = np.zeros(nrepeat)
            for iRep in range(nrepeat):
                t0 = time.perf_counter()
                acell.ring.finddyapsym4(
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    nx=nx,
                    ny=ny,
                    dp=0,
                    nturn=nturn,
                    dfu=False,
                    naf=False,
                    savetbt=False,
                )
                dts[iRep] = time.perf_counter() - t0
                print(f"dt = {dts[iRep]:.6f}")
            timing[pu_type][n] = dts

    with h5py.File(
        f"timeit_beamline_result_{datetime.now():%Y%m%dT%H%M%S}.h5", "w"
    ) as f:
        for pu_type, d in timing.items():
            g = f.create_group(pu_type)
            for n, v in d.items():
                g[str(n)] = v


def plot():

    timing = {}
    with h5py.File(f"timeit_beamline_result_20220823T064729.h5", "r") as f:
        for pu_type, g in f.items():
            timing[pu_type] = {}
            for n_str, v in g.items():
                timing[pu_type][int(n_str)] = v[()]

    import matplotlib.pyplot as plt

    plt.figure()
    ns = np.sort(list(timing["cpu"]))
    n_squares = ns**2
    vals = np.array([timing["cpu"][k] for k in ns])
    plt.loglog(n_squares, np.mean(vals, axis=1), "r.-", label="CPU")
    ns = np.sort(list(timing["gpu"]))
    n_squares = ns**2
    vals = np.array([timing["gpu"][k] for k in ns])
    plt.loglog(n_squares, np.mean(vals, axis=1), "b.-", label="GPU")
    plt.legend(loc="best")
    plt.xlabel(r"$\mathrm{Number\; of\; Particles}$", size="large")
    plt.ylabel(r"$\mathrm{Computation\; Time\; [s]}$", size="large")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    if sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "plot":
        plot()
