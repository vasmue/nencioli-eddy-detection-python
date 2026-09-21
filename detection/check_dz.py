#!/usr/bin/env python
"""
Find a suitable contour step dz for one day of data, independently of the
value in the config.

The starting step is taken from the data: the typical range of equivalent
surface elevation in the first window around a centre, divided into a few
levels.  The step is then halved until the results stop changing -- the
median radii and the number of centres that get a shape.  The recommended dz
is the coarsest step that already agrees with the next finer one; going finer
than that only costs time.

Usage:
    python check_dz.py UV_FILE CENTERS_FILE [--sample N]

    python check_dz.py uv_2015_001.nc eddy_centers_2015_001.nc
"""

import argparse
import time

import numpy as np

from nencioli_shapes import (
    MAX_GRID_LAT, eddy_shape, equivalent_elevation, remove_ramp,
    streamfunction,
)
from run_center_detection import read_coords, read_uv
from run_shape_detection import CFG, read_centers

# Two successive steps agree when their median radii differ by at most
# RADIUS_TOL percent and their numbers of shaped centres by at most SHAPED_TOL
# percent.
RADIUS_TOL = 5
SHAPED_TOL = 5

# Halvings tried at most, starting from the coarsest step.
MAX_HALVINGS = 7


def _round_down(x):
    """Largest 1, 2 or 5 times a power of ten not above x."""
    e = 10.0 ** np.floor(np.log10(x))
    return max(m * e for m in (1, 2, 5) if m * e <= x)


def main():
    shp = CFG["shape"]
    par = CFG["detection"]

    ap = argparse.ArgumentParser()
    ap.add_argument("uv_file")
    ap.add_argument("centers_file")
    ap.add_argument("--sample", type=int, default=200,
                    help="number of centres to shape (default 200)")
    args = ap.parse_args()

    u, v, date = read_uv(args.uv_file, par["input"])
    lon, lat = read_coords(args.uv_file, par["input"])
    dlon = float(lon[0, 1] - lon[0, 0])
    dlat = float(lat[1, 0] - lat[0, 0])
    nlat, nlon = u.shape

    rad = 2 * par["a"]
    facs = np.arange(1.0, shp["fac_max"], shp["fac_step"])
    radii = sorted({int(round(rad * f)) for f in facs})

    cen = read_centers(args.centers_file)
    ok = np.flatnonzero((np.abs(cen["lat"]) <= MAX_GRID_LAT) &
                        (cen["i"] >= rad) & (cen["i"] < nlat - rad) &
                        (cen["j"] >= rad) & (cen["j"] < nlon - rad))
    rng = np.random.default_rng(0)
    pick = rng.choice(ok, size=min(args.sample, ok.size), replace=False)

    # typical elevation range in the first window around a centre
    span = []
    for n in pick:
        i, j = int(cen["i"][n]), int(cen["j"][n])
        uw = u[i - rad:i + rad + 1, j - rad:j + rad + 1]
        vw = v[i - rad:i + rad + 1, j - rad:j + rad + 1]
        psi, ocean = streamfunction(uw, vw, rad, rad, float(cen["lat"][n]),
                                    dlon, dlat)
        psi = remove_ramp(psi, ocean, rad, rad, float(cen["lat"][n]),
                          dlon, dlat)
        eta = equivalent_elevation(psi, float(cen["true_lat"][n]))
        span.append(np.nanmax(eta) - np.nanmin(eta))
    span = float(np.median(span))

    # a handful of levels across the typical window to begin with
    dz0 = _round_down(span / 4)

    print(f"{date:%Y-%m-%d}: {pick.size} of {cen['i'].size} centres; "
          f"median elevation range in the first window {span:.2e} m")
    print(f"(configured dz is {shp['dz']} m, not used here)")

    def shape_all(dz):
        return [eddy_shape(u, v, int(cen["i"][n]), int(cen["j"][n]),
                           float(cen["lat"][n]), float(cen["true_lat"][n]),
                           dlon, dlat, radii, dz, shp["contour_points"],
                           shp["min_contours"]) for n in pick]

    def summary(run, common):
        res = [run[k] for k in common]
        med = lambda key: np.median([r[key] for r in res]) / 1e3
        return (sum(r is not None for r in run),
                med("effective_radius"), med("speed_radius"))

    # one pass first, so the timings below leave out building the cached
    # operators, as in a long run
    shape_all(dz0)

    print("\nradii are medians over all shaped centres; the comparison with "
          "the next finer step\nuses only the centres shaped at both, so the "
          "two can differ")
    print(f"{'dz (m)':>10} {'shaped':>7} {'contours':>9} {'r_eff km':>9} "
          f"{'r_speed km':>11} {'ms/eddy':>8}   vs. next finer step")

    best = None
    prev = None
    for k in range(MAX_HALVINGS + 1):
        dz = dz0 / 2 ** k
        t0 = time.time()
        run = shape_all(dz)
        ms = 1000 * (time.time() - t0) / pick.size

        shaped = [r for r in run if r is not None]
        con = np.median([r["n_contours"] for r in shaped]) if shaped else 0
        med_e = np.median([r["effective_radius"] for r in shaped]) / 1e3 \
            if shaped else np.nan
        med_s = np.median([r["speed_radius"] for r in shaped]) / 1e3 \
            if shaped else np.nan
        line = (f"{dz:>10.2e} {len(shaped):>7} {con:>9.0f} {med_e:>9.2f} "
                f"{med_s:>11.2f} {ms:>8.1f}")

        if prev is not None:
            # compare with the coarser step on the centres both could shape
            common = [m for m in range(pick.size)
                      if run[m] is not None and prev["run"][m] is not None]
            n1, e1, s1 = summary(prev["run"], common)
            n2, e2, s2 = summary(run, common)
            d_e = 100 * (e1 - e2) / e2
            d_s = 100 * (s1 - s2) / s2
            d_n = 100 * (n1 - n2) / n2
            prev["line"] += (f"   radii {d_e:+5.1f}% / {d_s:+5.1f}%, "
                             f"shaped {d_n:+5.1f}%")
            print(prev["line"])
            if (best is None and abs(d_e) <= RADIUS_TOL
                    and abs(d_s) <= RADIUS_TOL and -d_n <= SHAPED_TOL):
                best = prev["dz"]
                print(line)
                break

        prev = dict(dz=dz, run=run, line=line)
    else:
        print(prev["line"])

    if best is None:
        print(f"\nno recommendation: the results were still changing after "
              f"{MAX_HALVINGS} halvings, down to dz = {prev['dz']:.2e} m")
    else:
        print(f"\nrecommended dz: {best:.2e} m -- halving it changes the "
              f"radii by at most {RADIUS_TOL}% and the number of shaped "
              f"centres by at most {SHAPED_TOL}%")


if __name__ == "__main__":
    main()