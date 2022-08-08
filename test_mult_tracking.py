import time

import h5py
import numpy as np

import pylatt as latt

if __name__ == "__main__":

    quad = latt.quad("Q1", L=1, K1=0.5, nkick=20)

    test_dataset_filepath = "test_dataset_20220808T173734.hdf5"

    with h5py.File(test_dataset_filepath, "r") as f:
        ini_6d = f["ini_6d"][()]
        correct_fin_6d = f["fin_6d"][()]

    for _ in range(5):
        t0 = time.perf_counter()
        output = quad.sympass4(ini_6d, fast=1)
        print(f"Took {time.perf_counter()-t0:.3f} [s]")

        np.testing.assert_almost_equal(output, correct_fin_6d, decimal=16)
