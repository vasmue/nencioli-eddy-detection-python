#!/usr/bin/env python
"""
Run the Nencioli center detection over a set of gridded velocity files.

Works with any lon/lat gridded u/v: set the variable names in the config, and
say whether the grid is rotated.  Days are independent, so each is handled by
its own worker and writes its own file.

Usage:
    python run_center_detection.py FILE [FILE ...] [--out DIR]
                                   [--workers N] [--overwrite]

    python run_center_detection.py /path/2015/*.nc --workers 8
"""

import argparse
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num, num2date

from detection_config import load_detection_config
from nencioli_centers import (
    build_region_mask, rotated_to_geographic, uv_search,
)

CFG = load_detection_config()

_W = {}


def _grid_dims(ds, inp):
    """Names of the latitude and longitude dimensions.

    One-dimensional coordinates name their own dimension.  Two-dimensional
    ones share the same pair, so they are told apart by which of the two
    latitude varies along.  Neither test looks at array lengths, which say
    nothing when the grid is square.
    """
    latv, lonv = ds[inp["lat_var"]], ds[inp["lon_var"]]
    if latv.ndim == 1:
        return latv.dims[0], lonv.dims[0]

    d0, d1 = latv.dims
    a = np.asarray(latv.values, dtype=np.float64)
    if np.abs(np.diff(a, axis=0)).mean() >= np.abs(np.diff(a, axis=1)).mean():
        return d0, d1
    return d1, d0


def read_uv(path, inp):
    """u, v as 2-D (lat, lon), plus the timestamp."""
    with xr.open_dataset(path, decode_times=False) as ds:
        lat_dim, lon_dim = _grid_dims(ds, inp)
        u = np.asarray(ds[inp["u_var"]].squeeze()
                       .transpose(lat_dim, lon_dim).values)
        v = np.asarray(ds[inp["v_var"]].squeeze()
                       .transpose(lat_dim, lon_dim).values)

        tvar = ds[inp["time_var"]]
        t = np.asarray(tvar.values).ravel()[0]

        units = tvar.attrs.get("units", tvar.attrs.get("unit"))
        if units is None:
            raise ValueError(
                f"{os.path.basename(path)}: time variable "
                f"'{inp['time_var']}' has no units attribute")

        cal = tvar.attrs.get("calendar", "standard")
        date = num2date(
            t, units, calendar=cal,
            only_use_cftime_datetimes=False
        )

    return u, v, date


def read_date(path, inp):
    """Read only the timestamp from a file."""
    with xr.open_dataset(path, decode_times=False) as ds:
        tvar = ds[inp["time_var"]]
        t = np.asarray(tvar.values).ravel()[0]

        units = tvar.attrs.get("units", tvar.attrs.get("unit"))
        if units is None:
            raise ValueError(
                f"{os.path.basename(path)}: time variable "
                f"'{inp['time_var']}' has no units attribute")

        cal = tvar.attrs.get("calendar", "standard")

    return num2date(
        t, units, calendar=cal,
        only_use_cftime_datetimes=False
    )


def read_coords(path, inp):
    """Grid longitude and latitude as 2-D (lat, lon)."""
    with xr.open_dataset(path, decode_times=False) as ds:
        lat_dim, lon_dim = _grid_dims(ds, inp)
        latv, lonv = ds[inp["lat_var"]], ds[inp["lon_var"]]

        if latv.ndim == 1:
            lon, lat = np.meshgrid(np.asarray(lonv.values, dtype=np.float64),
                                   np.asarray(latv.values, dtype=np.float64))
        else:
            lon = np.asarray(lonv.transpose(lat_dim, lon_dim).values,
                             dtype=np.float64)
            lat = np.asarray(latv.transpose(lat_dim, lon_dim).values,
                             dtype=np.float64)
    return lon, lat


def geographic_coords(path, lon, lat, par):
    """Geographic coordinates for every grid cell.

    Three cases, in order of preference: the file already carries them, the
    grid is rotated and the angles are known, or the grid coordinates are
    themselves geographic.
    """
    with xr.open_dataset(path, decode_times=False) as ds:
        if "true_lon" in ds and "true_lat" in ds:
            lat_dim, lon_dim = _grid_dims(ds, par["input"])
            tlon = np.asarray(ds["true_lon"].transpose(lat_dim, lon_dim).values,
                              dtype=np.float64)
            tlat = np.asarray(ds["true_lat"].transpose(lat_dim, lon_dim).values,
                              dtype=np.float64)
            return tlon, tlat, "from file variables"

        attrs = ds.attrs
        angles = None
        if all(k in attrs for k in ("euler_alpha", "euler_beta", "euler_gamma")):
            angles = (attrs["euler_alpha"], attrs["euler_beta"],
                      attrs["euler_gamma"])
            src = f"file Euler attributes {angles}"

    if angles is None and par.get("rotated"):
        angles = tuple(par["euler"])
        src = f"config Euler angles {angles}"

    if angles is None:
        return lon.copy(), lat.copy(), "grid is already geographic"

    tlon, tlat = rotated_to_geographic(*angles, lon, lat)
    return tlon, tlat, src


def _init_worker(lon, lat, true_lon, true_lat, excl, par, out_dir):
    _W.update(lon=lon, lat=lat, true_lon=true_lon, true_lat=true_lat,
              excl=excl, par=par, out_dir=out_dir)


def _out_name(out_dir, date):
    return os.path.join(
        out_dir, f"eddy_centers_{date.year}_{date.timetuple().tm_yday:03d}.nc")


def _process(path):
    par = _W["par"]
    u, v, date = read_uv(path, par["input"])
    res = uv_search(u, v, _W["lon"], _W["lat"], _W["excl"], par["a"], par["b"])

    # counter-clockwise is cyclonic in the northern hemisphere and
    # anticyclonic in the southern
    tlat = _W["true_lat"][res["i"], res["j"]]
    res["type"] = np.where(tlat < 0, -res["sense"], res["sense"]).astype(np.int32)
    res["true_lat"] = tlat
    res["true_lon"] = _W["true_lon"][res["i"], res["j"]]

    _write(_out_name(_W["out_dir"], date), res, date, par)
    return date, res["i"].size


def _write(outfile, res, date, par):
    n = res["i"].size
    tmp = outfile + ".tmp"

    with Dataset(tmp, "w", format="NETCDF4") as f:
        f.createDimension("obs", n)

        def var(name, dtype, data, **attrs):
            v = f.createVariable(name, dtype, ("obs",), zlib=True, complevel=1)
            if n:
                v[:] = data
            for k, val in attrs.items():
                setattr(v, k, val)

        var("lat", "f8", res["lat"], units="degrees_north",
            long_name="eddy centre latitude on the model grid")
        var("lon", "f8", res["lon"], units="degrees_east",
            long_name="eddy centre longitude on the model grid")
        var("true_lat", "f8", res["true_lat"], units="degrees_north",
            long_name="eddy centre geographic latitude")
        var("true_lon", "f8", res["true_lon"], units="degrees_east",
            long_name="eddy centre geographic longitude")
        var("sense", "i4", res["sense"], long_name="rotation sense",
            comment="1 = counter-clockwise, -1 = clockwise")
        var("type", "i4", res["type"], long_name="eddy type",
            comment="1 = cyclonic, -1 = anticyclonic")
        var("i", "i4", res["i"], long_name="latitude index into the grid")
        var("j", "i4", res["j"], long_name="longitude index into the grid")

        tv = f.createVariable("time", "f8", ())
        tv.units = CFG["output"]["time_units"]
        tv.calendar = CFG["output"]["calendar"]
        tv.assignValue(date2num(date, tv.units, calendar=tv.calendar))

        f.day = date.strftime("%Y-%m-%d")
        f.a = par["a"]
        f.b = par["b"]
        f.algorithm = "Nencioli et al. (2010) vector geometry"

    os.replace(tmp, outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="gridded u/v files; shell wildcards work")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    par = dict(CFG["detection"])

    files = []
    for p in args.inputs:
        # already-expanded names pass through; a quoted wildcard still works
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    files = sorted(set(files))

    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"no such file: {missing[0]}")
    if not files:
        raise SystemExit("no input files")

    out_dir = args.out or par["centers_out"]
    os.makedirs(out_dir, exist_ok=True)

    lon, lat = read_coords(files[0], par["input"])
    true_lon, true_lat, src = geographic_coords(files[0], lon, lat, par)
    print(f"grid {lon.shape}, geographic coordinates {src}")
    print(f"geographic latitude range {true_lat.min():.2f} .. "
          f"{true_lat.max():.2f}")

    excl = ~build_region_mask(true_lon, true_lat, par["regions"])
    print(f"searching {int((~excl).sum())} of {excl.size} cells")
    print(f"detecting eddies in region "
          f"{true_lon[~excl].min():.2f} .. {true_lon[~excl].max():.2f} and "
          f"{true_lat[~excl].min():.2f} .. {true_lat[~excl].max():.2f}")

    todo = files
    if not args.overwrite:
        todo = []
        for p in files:
            date = read_date(p, par["input"])
            if not os.path.isfile(_out_name(out_dir, date)):
                todo.append(p)
    print(f"{len(todo)} of {len(files)} files to do, "
          f"a={par['a']} b={par['b']}, {args.workers} worker(s)")
    if not todo:
        return

    init = (lon, lat, true_lon, true_lat, excl, par, out_dir)

    t0 = time.time()
    total = 0
    if args.workers <= 1:
        _init_worker(*init)
        for k, p in enumerate(todo, 1):
            date, n = _process(p)
            total += n
            print(f"  {date:%Y-%m-%d}: {n} centres ({k}/{len(todo)})")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=init) as pool:
            for k, (date, n) in enumerate(
                    pool.imap_unordered(_process, todo), 1):
                total += n
                print(f"  {date:%Y-%m-%d}: {n} centres ({k}/{len(todo)})",
                      flush=True)

    dt = time.time() - t0
    print(f"{len(todo)} files in {dt:.1f} s ({dt/len(todo):.2f} s each), "
          f"{total} centres")


if __name__ == "__main__":
    main()