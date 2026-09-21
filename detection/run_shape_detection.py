#!/usr/bin/env python
"""
Compute eddy shapes for the centers found by run_center_detection.py.

Takes the same velocity files as the center stage.  Each file's date names
the centers file it is paired with, so the two stages line up without any
filename parsing.  Days are independent, so each is handled by its own
worker and writes its own file.

Usage:
    python run_shape_detection.py FILE [FILE ...] --centers DIR [--out DIR]
                                  [--workers N] [--overwrite]

    python run_shape_detection.py /path/2015/*.nc --centers CENTERS --workers 8
"""

import argparse
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num

from detection_config import load_detection_config
from nencioli_shapes import MAX_GRID_LAT, eddy_shape, nested
from run_center_detection import (
    geographic_coords, read_coords, read_date, read_uv,
)

CFG = load_detection_config()

_W = {}


def _day_name(prefix, out_dir, date):
    return os.path.join(
        out_dir, f"{prefix}_{date.year}_{date.timetuple().tm_yday:03d}.nc")


def read_centers(path):
    """Centers of one day."""
    with xr.open_dataset(path, decode_times=False) as ds:
        return {k: np.asarray(ds[k].values) for k in
                ("lat", "lon", "true_lat", "true_lon", "sense", "type",
                 "i", "j")}


def _init_worker(lon, lat, par, shp, radii, centers_dir, out_dir):
    _W.update(lon=lon, lat=lat, par=par, shp=shp, radii=radii,
              centers_dir=centers_dir, out_dir=out_dir)


def _process(path):
    par, shp = _W["par"], _W["shp"]
    date = read_date(path, par["input"])

    cen = read_centers(_day_name("eddy_centers", _W["centers_dir"], date))
    u, v, _ = read_uv(path, par["input"])

    dlon = float(_W["lon"][0, 1] - _W["lon"][0, 0])
    dlat = float(_W["lat"][1, 0] - _W["lat"][0, 0])
    npts = shp["contour_points"]

    keep, shapes = [], []
    for n in range(cen["i"].size):
        if abs(cen["lat"][n]) > MAX_GRID_LAT:
            continue
        res = eddy_shape(u, v, int(cen["i"][n]), int(cen["j"][n]),
                         float(cen["lat"][n]), float(cen["true_lat"][n]),
                         dlon, dlat, _W["radii"], shp["dz"], npts,
                         shp["min_contours"])
        if res is None:
            continue
        keep.append(n)
        shapes.append(res)

    if shapes:
        inner = nested(shapes, cen["i"][keep], cen["j"][keep],
                       cen["type"][keep])
        for s, f in zip(shapes, inner):
            s["nested"] = f

    # contour vertices from fractional grid indices to grid coordinates
    for s in shapes:
        for name in ("effective", "speed"):
            s[name + "_lon"] = _W["lon"][0, 0] + s[name + "_col"] * dlon
            s[name + "_lat"] = _W["lat"][0, 0] + s[name + "_row"] * dlat

    for s in shapes:
        s["too_small"] = s["effective_radius"] < CFG["tracking"]["min_rad"]

    keep = np.array(keep, dtype=np.int64)
    _write(_day_name("eddy_shapes", _W["out_dir"], date),
           cen, keep, shapes, date, par, shp)

    # the eddies to track: one file per type, as py-eddy-tracker expects
    kind = cen["type"][keep]
    usable = np.array([not s["nested"] and not s["large"] and
                       not s["too_small"] and
                       s["shape_error"] < CFG["tracking"]["max_err"]
                       for s in shapes], dtype=bool)
    for title, t in (("Cyclonic", 1), ("Anticyclonic", -1)):
        sel = np.flatnonzero(usable & (kind == t))
        _write_pet(_day_name(title, _W["out_dir"], date), title,
                   cen, keep[sel], [shapes[k] for k in sel], date, npts)

    n_con = np.median([s["n_contours"] for s in shapes]) if shapes else 0.0
    return date, len(keep), cen["i"].size, n_con


def _report(date, ns, nc, n_con, lo, hi, k, total):
    """One line per day, with a warning when the contour step looks wrong
    for this data: too coarse leaves few levels per eddy, too fine only
    costs time."""
    line = (f"  {date:%Y-%m-%d}: {ns} of {nc} centres shaped, "
            f"median {n_con:.0f} contours ({k}/{total})")
    if n_con < lo:
        line += f"  !! fewer than {lo}: dz is probably too coarse"
    elif n_con > hi:
        line += f"  !! more than {hi}: dz is probably finer than needed"
    print(line, flush=True)


def _write_pet(outfile, title, cen, keep, shapes, date, npts):
    """Eddies of one type in py-eddy-tracker's observation format.

    Positions and contours are in the coordinates of the model grid.  For a
    rotated grid that is still a sphere, and a rotation preserves distances
    and areas, so the tracking links the same eddies as it would in
    geographic coordinates.
    """
    n = keep.size
    tmp = outfile + ".tmp"

    with Dataset(tmp, "w", format="NETCDF4") as f:
        f.createDimension("obs", n)
        f.createDimension("NbSample", npts)

        def var(name, dtype, data, dims=("obs",), **attrs):
            v = f.createVariable(name, dtype, dims, zlib=True, complevel=1)
            if n:
                v[:] = data
            for k, val in attrs.items():
                setattr(v, k, val)

        pull = lambda k: np.array([s[k] for s in shapes]) if n else np.empty(0)

        units = "days since 1950-01-01 00:00:00"
        var("time", "f8", np.full(n, date2num(date, units, calendar="standard")),
            units=units, calendar="standard")
        var("longitude", "f4", cen["lon"][keep], units="degrees_east")
        var("latitude", "f4", cen["lat"][keep], units="degrees_north")
        var("amplitude", "f4", pull("amplitude"), units="m")
        var("effective_radius", "f4", pull("effective_radius"), units="m")
        var("speed_radius", "f4", pull("speed_radius"), units="m")
        var("speed_average", "f4", pull("speed"), units="m/s")
        var("effective_contour_shape_error", "f4", pull("shape_error"),
            units="%")
        for name in ("effective", "speed"):
            var(f"{name}_contour_longitude", "f4", pull(name + "_lon"),
                ("obs", "NbSample"))
            var(f"{name}_contour_latitude", "f4", pull(name + "_lat"),
                ("obs", "NbSample"))

        f.title = title

    os.replace(tmp, outfile)


def _write(outfile, cen, keep, shapes, date, par, shp):
    n = keep.size
    npts = shp["contour_points"]
    tmp = outfile + ".tmp"

    with Dataset(tmp, "w", format="NETCDF4") as f:
        f.createDimension("obs", n)
        f.createDimension("vertex", npts)

        def var(name, dtype, data, dims=("obs",), **attrs):
            v = f.createVariable(name, dtype, dims, zlib=True, complevel=1)
            if n:
                v[:] = data
            for k, val in attrs.items():
                setattr(v, k, val)

        for name, units, long_name in (
                ("lat", "degrees_north", "eddy centre latitude on the model grid"),
                ("lon", "degrees_east", "eddy centre longitude on the model grid"),
                ("true_lat", "degrees_north", "eddy centre geographic latitude"),
                ("true_lon", "degrees_east", "eddy centre geographic longitude")):
            var(name, "f8", cen[name][keep], units=units, long_name=long_name)

        var("sense", "i4", cen["sense"][keep], long_name="rotation sense",
            comment="1 = counter-clockwise, -1 = clockwise")
        var("type", "i4", cen["type"][keep], long_name="eddy type",
            comment="1 = cyclonic, -1 = anticyclonic")
        var("i", "i4", cen["i"][keep], long_name="latitude index into the grid")
        var("j", "i4", cen["j"][keep], long_name="longitude index into the grid")

        pull = lambda k: np.array([s[k] for s in shapes]) if n else np.empty(0)

        var("amplitude", "f8", pull("amplitude"), units="m",
            long_name="equivalent elevation difference between the centre "
                      "and the effective contour")
        var("effective_radius", "f8", pull("effective_radius"), units="m",
            long_name="radius of the circle with the area of the effective "
                      "contour")
        var("speed_radius", "f8", pull("speed_radius"), units="m",
            long_name="radius of the circle with the area of the speed "
                      "contour")
        var("speed", "f8", pull("speed"), units="m s-1",
            long_name="mean speed along the speed contour")
        var("shape_error", "f8", pull("shape_error"), units="percent",
            long_name="area between the effective contour and the circle of "
                      "equal area")
        var("n_contours", "i4", pull("n_contours"),
            long_name="closed contours found around the centre")
        var("weak_shape", "i4", pull("large").astype(np.int32),
            long_name="effective contour is only the largest closed contour",
            comment="1 = no contour passed the speed test: none had speed "
                    "increasing outward at all four extremes")
        var("nested", "i4", pull("nested").astype(np.int32),
            long_name="centre lies inside a larger eddy of the same type",
            comment="1 = secondary centre within another eddy; not to be "
                    "tracked")
        var("too_small", "i4", pull("too_small").astype(np.int32),
            long_name="effective radius below tracking.min_rad",
            comment="1 = too small to be a credible eddy; not to be tracked")
        var("window_radius", "i4", pull("rad"),
            long_name="half-width in grid points of the window the "
                      "streamfunction was solved on")

        for name in ("effective", "speed"):
            var(f"{name}_contour_lon", "f8", pull(name + "_lon"),
                ("obs", "vertex"), units="degrees_east",
                long_name=f"{name} contour longitude on the model grid")
            var(f"{name}_contour_lat", "f8", pull(name + "_lat"),
                ("obs", "vertex"), units="degrees_north",
                long_name=f"{name} contour latitude on the model grid")

        tv = f.createVariable("time", "f8", ())
        tv.units = CFG["output"]["time_units"]
        tv.calendar = CFG["output"]["calendar"]
        tv.assignValue(date2num(date, tv.units, calendar=tv.calendar))

        f.day = date.strftime("%Y-%m-%d")
        f.a = par["a"]
        f.b = par["b"]
        f.dz = shp["dz"]
        f.min_contours = shp["min_contours"]
        f.min_rad = CFG["tracking"]["min_rad"]
        f.algorithm = "Nencioli et al. (2010) vector geometry"

    os.replace(tmp, outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="gridded u/v files; shell wildcards work")
    ap.add_argument("--centers", required=True,
                    help="directory of eddy_centers files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    par = dict(CFG["detection"])
    if "shape" not in CFG:
        raise SystemExit("config has no 'shape' section")
    shp = dict(CFG["shape"])

    files = []
    for p in args.inputs:
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    files = sorted(set(files))

    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"no such file: {missing[0]}")
    if not files:
        raise SystemExit("no input files")

    out_dir = args.out or shp["shapes_out"]
    os.makedirs(out_dir, exist_ok=True)

    lon, lat = read_coords(files[0], par["input"])
    _, true_lat, src = geographic_coords(files[0], lon, lat, par)
    print(f"grid {lon.shape}, geographic coordinates {src}")

    rad = 2 * par["a"]
    facs = np.arange(1.0, shp["fac_max"], shp["fac_step"])
    radii = sorted({int(round(rad * f)) for f in facs})
    print(f"contour step {shp['dz']} m, windows {radii}")

    todo = []
    for p in files:
        date = read_date(p, par["input"])
        if not os.path.isfile(_day_name("eddy_centers", args.centers, date)):
            continue
        if args.overwrite or not os.path.isfile(
                _day_name("eddy_shapes", out_dir, date)):
            todo.append(p)
    print(f"{len(todo)} of {len(files)} files to do, {args.workers} worker(s)")
    if not todo:
        return

    init = (lon, lat, par, shp, radii, args.centers, out_dir)

    lo, hi = shp["contours_warn"]

    t0 = time.time()
    total = 0
    if args.workers <= 1:
        _init_worker(*init)
        for k, p in enumerate(todo, 1):
            date, ns, nc, n_con = _process(p)
            total += ns
            _report(date, ns, nc, n_con, lo, hi, k, len(todo))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=init) as pool:
            for k, (date, ns, nc, n_con) in enumerate(
                    pool.imap_unordered(_process, todo), 1):
                total += ns
                _report(date, ns, nc, n_con, lo, hi, k, len(todo))

    dt = time.time() - t0
    print(f"{len(todo)} files in {dt:.1f} s ({dt/len(todo):.2f} s each), "
          f"{total} shapes")


if __name__ == "__main__":
    main()