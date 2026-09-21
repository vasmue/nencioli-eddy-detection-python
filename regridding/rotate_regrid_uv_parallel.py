#!/usr/bin/env python
"""
Interpolate FESOM u/v at one depth level onto the rotated regular grid,
using weights precomputed by precompute_weights.py.

This script builds nothing expensive: it mmaps the weight arrays (so several
concurrent workers share one copy through the page cache) and each timestep is
a gather plus three multiplies.  Peak memory is roughly one year of the level
slice plus a 100 MB output buffer.

Usage:
    python rotate_regrid_uv.py <year> <depth_in_m> [weights_dir]
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num
from pandas import Timestamp

from config import check_meta_matches, load_config
from mesh_diag import load_mesh_diag

# ----------------------------------------------------------------------------
# configuration -- everything comes from config.yaml
# ----------------------------------------------------------------------------

CFG = load_config()

DATA_PATH = CFG["paths"]["data_path"]
MESH_PATH = CFG["paths"]["mesh_path"]
OUT_ROOT = CFG["paths"]["out_root"]
WEIGHTS_DIR = CFG["paths"]["weights_dir"]

TIME_UNITS = CFG["output"]["time_units"]
CALENDAR = CFG["output"]["calendar"]

OVERWRITE = CFG["output"]["overwrite"]
MIN_VALID_BYTES = CFG["output"]["min_valid_mb"] * 1024 * 1024
COMPLEVEL = CFG["output"]["complevel"]
CHUNKSIZES = (1, CFG["output"]["chunk_lon"], CFG["output"]["chunk_lat"])


def dry_mask(max_dep, depth, u, v, verbose=False):
    """True where the sea floor is shallower than this level.

    Older FESOM writes dry nodes as 0.0 rather than NaN, so the bathymetry is
    invisible to isfinite().  Comparing the level depth against the per-node
    bottom depth is unambiguous -- unlike a level *count*, there is no
    interface-vs-layer or 0-vs-1-based convention to get wrong.

    Falls back to the exact-zero test only if zbar_n_bottom is unavailable.
    """
    if max_dep is None:
        return (u == 0.0) & (v == 0.0)

    dry = depth > max_dep

    if verbose:
        # cross-check against the zeros the model actually wrote
        is_zero = (u == 0.0) & (v == 0.0)
        n_dry = int(dry.sum())
        if n_dry:
            print(f"    {n_dry} nodes below the floor, "
                  f"{100 * is_zero[dry].mean():.3f}% of them exactly zero")
        if (~dry).sum():
            print(f"    {100 * is_zero[~dry].mean():.3f}% of wet nodes "
                  f"exactly zero")

    return dry


class Weights:
    """Precomputed interpolation weights.

    The grid-sized arrays are memory-mapped, so concurrent workers share one
    copy through the page cache.  The node-sized rotation coefficients are
    read into memory since they need a dtype cast.
    """

    def __init__(self, path, cfg=None):
        with open(os.path.join(path, "meta.json")) as fh:
            self.meta = json.load(fh)

        # the point of the shared config: refuse to run if config.yaml has been
        # edited since these weights were built
        check_meta_matches(cfg or CFG, self.meta, path)

        def load(name):
            return np.load(os.path.join(path, name), mmap_mode="r")

        self.valid = load("valid.npy")
        self.v0 = load("v0.npy")
        self.v1 = load("v1.npy")
        self.v2 = load("v2.npy")
        self.w0 = load("w0.npy")
        self.w1 = load("w1.npy")
        self.lon_eq = np.load(os.path.join(path, "lon_eq.npy"))
        self.lat_eq = np.load(os.path.join(path, "lat_eq.npy"))
        # saved float64 (direction cosines, order 1); float32 costs ~1e-7
        # relative error on the velocities, far below interpolation error, and
        # keeps the whole node-array path single precision.  These are the
        # only mmapped arrays we materialise -- they are node-sized, not
        # grid-sized, so the copy is cheap.
        self.rot_a = load("rot_a.npy").astype(np.float32)
        self.rot_b = load("rot_b.npy").astype(np.float32)
        self.rot_c = load("rot_c.npy").astype(np.float32)
        self.rot_d = load("rot_d.npy").astype(np.float32)

        self.nx = self.meta["nx"]
        self.ny = self.meta["ny"]

    def rotate(self, u, v):
        """Mesh-native vector frame -> rotated-grid frame, in one linear step."""
        return (self.rot_a * u + self.rot_b * v,
                self.rot_c * u + self.rot_d * v)

    def interpolate(self, field):
        """Gather + barycentric weighting onto the regular grid.

        Identical to matplotlib's LinearTriInterpolator, but with the triangle
        search already done.  NaN at any vertex propagates to the target cell,
        which is what we want: it keeps cells below the local bathymetry masked.
        """
        w0 = self.w0
        w1 = self.w1
        w2 = np.float32(1.0) - w0 - w1

        vals = (w0 * field[self.v0]
                + w1 * field[self.v1]
                + w2 * field[self.v2])

        out = np.full((self.nx, self.ny), np.nan, dtype=np.float32)
        out[self.valid] = vals
        return out


def write_netcdf(path, wts, u_reg, v_reg, when, depth_m):
    tmp = path + ".tmp"
    with Dataset(tmp, "w", format="NETCDF4") as fw:
        fw.createDimension("TIME", 1)
        fw.createDimension("LATITUDE", wts.ny)
        fw.createDimension("LONGITUDE", wts.nx)

        lat = fw.createVariable("LATITUDE", "f4", ("LATITUDE",))
        lon = fw.createVariable("LONGITUDE", "f4", ("LONGITUDE",))
        time = fw.createVariable("TIME", "f8", ("TIME",))

        kw = dict(zlib=True, complevel=COMPLEVEL, shuffle=True,
                  fill_value=np.float32(np.nan),
                  chunksizes=CHUNKSIZES)
        u = fw.createVariable("u", "f4", ("TIME", "LONGITUDE", "LATITUDE"), **kw)
        v = fw.createVariable("v", "f4", ("TIME", "LONGITUDE", "LATITUDE"), **kw)

        # NOTE: whether these coordinates are geographic depends on the mode
        # the weights were built in.  The units below are what the original
        # files carried and are kept for compatibility; the standard names and
        # the global Euler angles are what actually define the frame.  When the
        # frame is rotated, anything computing f or eddy polarity must use the
        # true coordinates, not these.
        rotated = wts.meta["mode"] == "rotate_regrid"
        frame = "rotated" if rotated else "geographic"

        lat.units = "degrees_north"
        lat.long_name = f"{frame} latitude"
        lat.standard_name = "grid_latitude" if rotated else "latitude"
        lon.units = "degrees_east"
        lon.long_name = f"{frame} longitude"
        lon.standard_name = "grid_longitude" if rotated else "longitude"

        time.units = TIME_UNITS
        time.calendar = CALENDAR
        time.long_name = "time"

        u.units = "m/s"
        u.long_name = f"eastward velocity in {frame} frame"
        v.units = "m/s"
        v.long_name = f"northward velocity in {frame} frame"

        lat[:] = wts.lat_eq
        lon[:] = wts.lon_eq
        time[:] = date2num(when, TIME_UNITS, calendar=CALENDAR)
        u[0] = u_reg
        v[0] = v_reg

        m = wts.meta
        fw.coordinate_frame = m["coordinate_frame"]
        fw.euler_alpha = m["euler_alpha"]
        fw.euler_beta = m["euler_beta"]
        fw.euler_gamma = m["euler_gamma"]
        fw.euler_convention = (
            "pyfesom2 scalar_g2r/scalar_r2g; apply scalar_r2g(alpha, beta, "
            "gamma, lon, lat) to recover geographic coordinates"
        )
        fw.true_latitude_positive = "yes" if m["bottom"] >= 0 else "no"
        fw.depth = float(depth_m)
        fw.mesh_path = m["mesh_path"]
        fw.source = "FESOM2, regridded with precomputed barycentric weights"

    os.replace(tmp, path)


_W = {}


def _init_worker(weights_dir, ufile, vfile, iz, z_actual, max_dep, out_path,
                 yy):
    """Runs once per worker process.

    The netCDF handles are opened HERE, not in the parent: HDF5 file handles
    inherited across a fork are shared state and corrupt or hang.  The mmapped
    weight arrays are the opposite case -- they are inherited harmlessly and
    all workers read the same physical pages via the page cache.
    """
    _W["wts"] = Weights(weights_dir)
    _W["unod"] = xr.open_dataset(ufile).unod.isel(nz1=iz)
    _W["vnod"] = xr.open_dataset(vfile).vnod.isel(nz1=iz)
    _W["iz"] = iz
    _W["z_actual"] = z_actual
    _W["max_dep"] = max_dep
    _W["out_path"] = out_path
    _W["yy"] = yy


def _process_day(ii):
    wts = _W["wts"]
    z_actual = _W["z_actual"]

    t = Timestamp(_W["unod"].coords["time"].values[ii]).to_pydatetime()
    doy = t.timetuple().tm_yday
    outfile = os.path.join(
        _W["out_path"],
        f"uv_gridded_{_W['yy']}_{doy:03d}_{z_actual}m.nc",
    )

    u_nat = _W["unod"].isel(time=ii).values.astype(np.float32, copy=False)
    v_nat = _W["vnod"].isel(time=ii).values.astype(np.float32, copy=False)

    # Older FESOM writes dry nodes as 0.0, not NaN.  Left in place, those zeros
    # get blended into every cell within one element of the sea floor,
    # producing shear and speed minima that are pure artefact -- exactly the
    # signature the Nencioli constraints look for.
    dry = dry_mask(_W["max_dep"], z_actual, u_nat, v_nat)

    nan32 = np.float32(np.nan)
    bad = dry | ~np.isfinite(u_nat) | ~np.isfinite(v_nat)
    u_nat = np.where(bad, nan32, u_nat)
    v_nat = np.where(bad, nan32, v_nat)

    u_rot, v_rot = wts.rotate(u_nat, v_nat)
    write_netcdf(outfile, wts, wts.interpolate(u_rot), wts.interpolate(v_rot),
                 t, z_actual)
    return doy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("depth", type=int)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes pooled over days within the year")
    ap.add_argument("--weights", default=WEIGHTS_DIR)
    args = ap.parse_args()

    out_path = os.path.join(OUT_ROOT, str(args.year))
    os.makedirs(out_path, exist_ok=True)

    # only the vertical axis is needed here; nz1 comes back positive-down and
    # is the axis unod/vnod are actually indexed on, so no interface/midpoint
    # mismatch as with mesh.zlev
    _, _, _, nz1, max_dep, _ = load_mesh_diag(MESH_PATH)
    iz = int(np.argmin(np.abs(nz1 - args.depth)))
    z_actual = int(round(nz1[iz]))
    print(f"requested {args.depth} m -> nz1 index {iz} at {z_actual} m")

    ufile = os.path.join(DATA_PATH, f"unod.fesom.{args.year}.nc")
    vfile = os.path.join(DATA_PATH, f"vnod.fesom.{args.year}.nc")

    # decide the work list in the parent, so workers never race on the same
    # output file and the existing resume logic is applied exactly once
    with xr.open_dataset(ufile) as ds:
        times = ds.time.values
    todo = []
    for ii in range(times.shape[0]):
        t = Timestamp(times[ii]).to_pydatetime()
        doy = t.timetuple().tm_yday
        outfile = os.path.join(
            out_path,
            f"uv_gridded_{args.year}_{doy:03d}_{z_actual}m.nc")
        if os.path.isfile(outfile):
            if os.path.getsize(outfile) < MIN_VALID_BYTES:
                print(f"  day {doy}: corrupted, removing")
                os.remove(outfile)
            elif not OVERWRITE:
                continue
        todo.append(ii)

    print(f"{len(todo)} of {times.shape[0]} days to do, "
          f"{args.workers} worker(s)")
    if not todo:
        return

    init_args = (args.weights, ufile, vfile, iz, z_actual, max_dep,
                 out_path, args.year)

    t0 = time.time()
    if args.workers <= 1:
        _init_worker(*init_args)
        for n, ii in enumerate(todo, 1):
            doy = _process_day(ii)
            print(f"  {args.year} {z_actual}m / day {doy:03d} done "
                  f"({n}/{len(todo)})")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=init_args) as pool:
            for n, doy in enumerate(pool.imap_unordered(_process_day, todo), 1):
                print(f"  {args.year} {z_actual}m / day {doy:03d} done "
                      f"({n}/{len(todo)})", flush=True)

    dt = time.time() - t0
    print(f"{len(todo)} days in {dt:.1f} s ({dt / len(todo):.2f} s/day)")


if __name__ == "__main__":
    main()