from datetime import datetime

import h5py
import numpy as np

import pylatt as latt


def generate_test_input_data():

    rng = np.random.default_rng(seed=42)

    nparticles = 1_000_000

    transv = rng.uniform(low=-10e-3, high=+10e-3, size=(4, nparticles))
    dp = rng.uniform(low=-5e-2, high=+5e-2, size=(1, nparticles))

    return np.vstack((transv, np.zeros((1, nparticles)), dp))


if __name__ == "__main__":

    quad = latt.quad("Q1", L=1, K1=0.5, nkick=20)

    x0 = generate_test_input_data()

    output = quad.sympass4(x0, fast=1)

    with h5py.File(f"test_dataset_{datetime.now():%Y%m%dT%H%M%S}.hdf5", "w") as f:
        f.create_dataset("ini_6d", data=x0, compression="gzip")
        f.create_dataset("fin_6d", data=output, compression="gzip")
