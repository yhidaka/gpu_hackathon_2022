import os
import sys
import time

# Must set these before NumPy import to disable its multithreading
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1

import cupy as cp
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import nvtx

import pylatt as latt

if __name__ == "__main__":

    if sys.argv[1] == "gpu":
        use_gpu = True
    elif sys.argv[1] == "cpu":
        use_gpu = False
    else:
        raise ValueError

    if use_gpu:
        latt.use_gpu()
    else:
        latt.use_cpu()

    import nsls2sr_supercell as acell

    with nvtx.annotate("First DA", color="red"):
        t0 = time.perf_counter()
        acell.ring.finddyapsym4(
            xmin=-0.05,
            xmax=0.04,
            ymin=1e-6,
            ymax=0.02,
            nx=31,
            ny=11,
            dp=0,
            nturn=256,
            dfu=False,
            naf=False,
        )
        print(f"DA took {time.perf_counter()-t0:.6f}")

    acell.ring.pltdyap()

    sh1 = acell.ring.getElements("sext", "sh1", unique=True)[0]
    sh1.K2 = 12

    with nvtx.annotate("Second DA", color="red"):
        t0 = time.perf_counter()
        acell.ring.finddyapsym4(
            xmin=-0.05,
            xmax=0.04,
            ymin=1e-6,
            ymax=0.02,
            nx=31,
            ny=11,
            dp=0,
            nturn=256,
            dfu=False,
            naf=False,
        )
        print(f"DA took {time.perf_counter()-t0:.6f}")

    acell.ring.pltdyap()

    pp = PdfPages(".".join(__file__.split(".")[:-1] + ["pdf"]))
    for fignum in plt.get_fignums():
        pp.savefig(figure=fignum)
    pp.close()

    # plt.show()
