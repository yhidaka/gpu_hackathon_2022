from datetime import datetime
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


def generate_test_input_data(use_gpu, nx, ny, nLattices):

    xmin = -1e-3
    xmax = +1e-3
    ymin = 1e-6
    ymax = 2e-3
    nturn = 256

    import nsls2sr_supercell as acell

    sexts = []
    for elem in acell.ring.bl:
        if isinstance(elem, latt.sext):
            print(elem.name)
            sexts.append(elem)

    nSexts = len(sexts)
    assert nSexts == 270 // 15

    rng = np.random.default_rng(seed=42)
    K2s = rng.uniform(low=-50, high=+50, size=(nLattices, nSexts))

    dyap_list = []
    fin_coords_list = []
    xgrid_list = []
    ygrid_list = []

    for iLat, new_K2_list in enumerate(K2s):

        print(f"{iLat+1}/{nLattices}")
        t0 = time.perf_counter()

        for elem, K2 in zip(sexts, new_K2_list):
            elem.K2 = K2
            elem._update()

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
            save_fin_coords=True,
        )
        dyap_list.append(acell.ring.dyap["dyap"])
        fin_coords_list.append(acell.ring.dyap["fin_coords"])
        xgrid_list.append(acell.ring.dyap["xgrid"])
        ygrid_list.append(acell.ring.dyap["ygrid"])

        print(f"dt = {time.perf_counter()-t0:.3f}")

    # Reshape to [nLattices, ny, nx]
    xgrid = np.moveaxis(np.dstack(xgrid_list), 2, 0)
    ygrid = np.moveaxis(np.dstack(ygrid_list), 2, 0)
    dyap = np.moveaxis(np.dstack(dyap_list), 2, 0)
    fin_x = np.moveaxis(
        np.dstack([v[0].reshape(ny, nx) for v in fin_coords_list]), 2, 0
    )
    fin_y = np.moveaxis(
        np.dstack([v[2].reshape(ny, nx) for v in fin_coords_list]), 2, 0
    )

    if use_gpu:
        pu_type = "gpu"
        xgrid = xgrid.get()
        ygrid = ygrid.get()
        dyap = dyap.get()
        fin_x = fin_x.get()
        fin_y = fin_y.get()
    else:
        pu_type = "cpu"

    with h5py.File(
        f"test_dataset_{pu_type}_x{nx}y{ny}lat{nLattices}_{datetime.now():%Y%m%dT%H%M%S}.hdf5",
        "w",
    ) as f:
        f.create_dataset("K2s", data=K2s, compression="gzip")
        f.create_dataset("xgrid", data=xgrid, compression="gzip")
        f.create_dataset("ygrid", data=ygrid, compression="gzip")
        f.create_dataset("dyap", data=dyap, compression="gzip")
        f.create_dataset("fin_x", data=fin_x, compression="gzip")
        f.create_dataset("fin_y", data=fin_y, compression="gzip")


def load_data_from_file(use_gpu):

    if False:
        if use_gpu:
            test_dataset_filepath = "test_dataset_gpu_x5y3lat2_20220823T131411.hdf5"
        else:
            test_dataset_filepath = "test_dataset_cpu_x5y3lat2_20220823T131516.hdf5"
    else:
        if use_gpu:
            test_dataset_filepath = (
                "test_dataset_gpu_x100y100lat10_20220823T134405.hdf5"
            )
        else:
            test_dataset_filepath = (
                "test_dataset_cpu_x100y100lat10_20220823T135233.hdf5"
            )

    with h5py.File(test_dataset_filepath, "r") as f:
        K2s = f["K2s"][()]
        xgrid = f["xgrid"][()]
        ygrid = f["ygrid"][()]
        correct_dyap = f["dyap"][()]
        correct_fin_x = f["fin_x"][()]
        correct_fin_y = f["fin_y"][()]

    return K2s, xgrid, ygrid, correct_dyap, correct_fin_x, correct_fin_y


def to_gpu(x):

    t0 = time.perf_counter()
    x_in_gpu = cp.asarray(x)
    print(f"CPU-to-GPU took {time.perf_counter()-t0:.3f} [s]")

    return x_in_gpu


if __name__ == "__main__":

    if sys.argv[1] == "gpu:single_lat":
        use_gpu = True
        multi_lat = False
    elif sys.argv[1] == "cpu:single_lat":
        use_gpu = False
        multi_lat = False
    elif sys.argv[1] == "gpu:multi_lat":
        use_gpu = True
        multi_lat = True
    elif sys.argv[1] == "cpu:multi_lat":
        use_gpu = False
        multi_lat = True
    elif sys.argv[1] == "gen_test_data":
        if sys.argv[2] == "cpu":
            use_gpu = False
        elif sys.argv[2] == "gpu":
            use_gpu = True
        else:
            raise ValueError
        multi_lat = False
    else:
        raise ValueError

    if use_gpu:
        latt.use_gpu()
    else:
        latt.use_cpu()

    if sys.argv[1] == "gen_test_data":
        if False:
            nx = 5
            ny = 3
            nLattices = 2
        else:
            nx = 100
            ny = 100
            nLattices = 10
        generate_test_input_data(use_gpu, nx, ny, nLattices)
        sys.exit(0)

    with nvtx.annotate("Load data from file", color="blue"):
        (
            K2s,
            corr_xgrid,
            corr_ygrid,
            corr_dyap,
            corr_fin_x,
            corr_fin_y,
        ) = load_data_from_file(use_gpu)
        # K2s: [nLattices, nSexts]
        # corr_xgrid, corr_ygrid, corr_dyap, corr_fin_x, corr_fin_y: [nLattices, ny, nx]

    if use_gpu:
        with nvtx.annotate("Transfer data to GPU", color="yellow"):
            corr_xgrid = to_gpu(corr_xgrid)
            corr_ygrid = to_gpu(corr_ygrid)
            corr_dyap = to_gpu(corr_dyap)
            corr_fin_x = to_gpu(corr_fin_x)
            corr_fin_y = to_gpu(corr_fin_y)

    nLattices, ny, nx = corr_dyap.shape

    xmin = -1e-3
    xmax = +1e-3
    ymin = 1e-6
    ymax = 2e-3
    nturn = 256

    import nsls2sr_supercell as acell

    sexts = []
    for elem in acell.ring.bl:
        if isinstance(elem, latt.sext):
            print(elem.name)
            sexts.append(elem)

    nSexts = len(sexts)
    assert nSexts == 270 // 15

    # decimal = 16
    decimal = 15

    with nvtx.annotate("Tracking", color="red"):

        t0 = time.perf_counter()

        if use_gpu:
            ncp = cp
        else:
            ncp = np

        if not multi_lat:
            dyap_list = []
            fin_coords_list = []
            xgrid_list = []
            ygrid_list = []

            for new_K2_list in K2s:
                for elem, K2 in zip(sexts, new_K2_list):
                    elem.K2 = K2
                    elem._update()

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
                    save_fin_coords=True,
                )
                dyap_list.append(acell.ring.dyap["dyap"])
                fin_coords_list.append(acell.ring.dyap["fin_coords"])
                xgrid_list.append(acell.ring.dyap["xgrid"])
                ygrid_list.append(acell.ring.dyap["ygrid"])

            # Reshape to [nLattices, ny, nx]
            xgrid = np.moveaxis(np.dstack(xgrid_list), 2, 0)
            ygrid = np.moveaxis(np.dstack(ygrid_list), 2, 0)
            dyap = np.moveaxis(np.dstack(dyap_list), 2, 0)
            fin_x = np.moveaxis(
                np.dstack([v[0].reshape(ny, nx) for v in fin_coords_list]), 2, 0
            )
            fin_y = np.moveaxis(
                np.dstack([v[2].reshape(ny, nx) for v in fin_coords_list]), 2, 0
            )

        else:
            for iSext, elem in enumerate(sexts):
                elem.K2_array = K2s[:, iSext]
                elem._update()

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
                save_fin_coords=True,
            )

            dyap = acell.ring.dyap["dyap"]
            fin_coords = acell.ring.dyap["fin_coords"]

            fin_x = fin_coords[0]
            fin_y = fin_coords[2]

            xgrid_list = [acell.ring.dyap["xgrid"] for _ in range(nLattices)]
            ygrid_list = [acell.ring.dyap["ygrid"] for _ in range(nLattices)]

            # Final array shapes will all be: [nLattices, ny, nx]
            fin_x = fin_x.reshape(nLattices, ny, nx)
            fin_y = fin_y.reshape(nLattices, ny, nx)
            xgrid = ncp.moveaxis(np.dstack(xgrid_list), 2, 0)
            ygrid = ncp.moveaxis(np.dstack(ygrid_list), 2, 0)

        print(f"Whole tracking took {time.perf_counter()-t0:.3f} [s]")

    with nvtx.annotate("Testing", color="green"):
        if use_gpu:
            cp.testing.assert_array_almost_equal(xgrid, corr_xgrid, decimal=decimal)
            cp.testing.assert_array_almost_equal(ygrid, corr_ygrid, decimal=decimal)
            cp.testing.assert_array_almost_equal(dyap, corr_dyap, decimal=decimal)
            cp.testing.assert_array_almost_equal(fin_x, corr_fin_x, decimal=decimal)
            cp.testing.assert_array_almost_equal(fin_y, corr_fin_y, decimal=decimal)
        else:
            np.testing.assert_almost_equal(xgrid, corr_xgrid, decimal=decimal)
            np.testing.assert_almost_equal(ygrid, corr_ygrid, decimal=decimal)
            np.testing.assert_almost_equal(dyap, corr_dyap, decimal=decimal)
            np.testing.assert_almost_equal(fin_x, corr_fin_x, decimal=decimal)
            np.testing.assert_almost_equal(fin_y, corr_fin_y, decimal=decimal)

    print("Finished successfully")
