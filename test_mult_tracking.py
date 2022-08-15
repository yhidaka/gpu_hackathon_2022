import os
import sys
import time

# Must set these before NumPy import to disable its multithreading
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1

import cupy as cp
import h5py
import numpy as np
import nvtx

import pylatt as latt


def load_data_from_file():

    test_dataset_filepath = "test_dataset_20220808T173734.hdf5"

    with h5py.File(test_dataset_filepath, "r") as f:
        ini_6d = f["ini_6d"][()]
        correct_fin_6d = f["fin_6d"][()]

    return ini_6d, correct_fin_6d


def to_gpu(x):

    t0 = time.perf_counter()
    x_in_gpu = cp.asarray(x)
    print(f"CPU-to-GPU took {time.perf_counter()-t0:.3f} [s]")

    return x_in_gpu


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

    with nvtx.annotate("Quad creation", color="red"):
        quad = latt.quad("Q1", L=1, K1=0.5, nkick=20)

    with nvtx.annotate("Load data from file", color="blue"):
        ini_6d, correct_fin_6d = load_data_from_file()

    if use_gpu:
        with nvtx.annotate("Transfer data to GPU", color="yellow"):
            ini_6d = to_gpu(ini_6d)
            correct_fin_6d = to_gpu(correct_fin_6d)

    # decimal = 16
    decimal = 15

    for _ in range(5):
        with nvtx.annotate("Tracking through the quad", color="red"):
            t0 = time.perf_counter()
            output = quad.sympass4(ini_6d, fast=1)
            print(f"Tracking took {time.perf_counter()-t0:.3f} [s]")

        with nvtx.annotate("Testing", color="green"):
            if use_gpu:
                cp.testing.assert_array_almost_equal(
                    output, correct_fin_6d, decimal=decimal
                )
            else:
                np.testing.assert_almost_equal(output, correct_fin_6d, decimal=decimal)
