#!/usr/bin/env python
"""
One-off validation for the precomputed regridding.  Run once per
(mesh, target grid, Euler angles); nothing here needs repeating per year,
depth or timestep.

Three checks, in increasing order of what they can catch:

  A  weight self-consistency   -- proves the gather+weight path reproduces
                                  matplotlib's LinearTriInterpolator exactly,
                                  without building the trifinder again
  B  speed invariance          -- |V| is unchanged by a frame rotation
  C  vorticity cross-check     -- the only check with power over whether the
                                  rotation *angles* are right

Usage:
    python validate_regrid.py <year> <depth_m> [--day N] [--level-index K]
                                               [--weights DIR]
"""

import argparse
import os
import sys

import numpy as np
import pyfesom2 as pf
import xarray as xr
from pandas import Timestamp

from config import euler_angles, load_config
from mesh_diag import load_mesh_diag
from rotate_regrid_uv_parallel import Weights, dry_mask

CFG = load_config()

MODE = CFG["mode"]
if MODE == "none":
    raise SystemExit(
        "mode is none: the input is already on a lon/lat grid, so there is "
        "no regridding to validate")

DATA_PATH = CFG["paths"]["data_path"]
MESH_PATH = CFG["paths"]["mesh_path"]
WEIGHTS_DIR = CFG["paths"]["weights_dir"]

ALPHA, BETA, GAMMA = euler_angles(CFG)
MESH_ABG = CFG["rotation"]["mesh_abg"]

R_EARTH = 6371000.0

# how many valid points to use for the (cheap but memory-hungry) checks
SUBSAMPLE = 2_000_000


def hr(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# check A -- weight self-consistency
# ---------------------------------------------------------------------------

def check_weights(wts, lons_rot, lats_rot):
    """A point's barycentric coordinates reconstruct its own position, and
    they lie in [0, 1].  Triangles do not overlap, so a point that satisfies
    both is inside the triangle we assigned it to -- and linear interpolation
    is uniquely determined by barycentric coordinates.  Passing this check
    therefore *proves* equivalence to LinearTriInterpolator, with no need to
    build the trifinder again.
    """
    hr("A. weight self-consistency")

    nvalid = int(wts.valid.sum())
    step = max(1, nvalid // SUBSAMPLE)
    sl = slice(None, None, step)
    print(f"  checking {len(range(0, nvalid, step))} of {nvalid} points "
          f"(every {step})")

    v0 = np.asarray(wts.v0[sl])
    v1 = np.asarray(wts.v1[sl])
    v2 = np.asarray(wts.v2[sl])
    w0 = np.asarray(wts.w0[sl], dtype=np.float64)
    w1 = np.asarray(wts.w1[sl], dtype=np.float64)
    w2 = 1.0 - w0 - w1

    # target-cell coordinates for the same subset
    lon_eq = wts.lon_eq.astype(np.float64)
    lat_eq = wts.lat_eq.astype(np.float64)
    xx = np.broadcast_to(lon_eq[:, None], (wts.nx, wts.ny))
    yy = np.broadcast_to(lat_eq[None, :], (wts.nx, wts.ny))
    x_target = xx[wts.valid][sl]
    y_target = yy[wts.valid][sl]

    x_recon = w0 * lons_rot[v0] + w1 * lons_rot[v1] + w2 * lons_rot[v2]
    y_recon = w0 * lats_rot[v0] + w1 * lats_rot[v1] + w2 * lats_rot[v2]

    dx = np.abs(x_recon - x_target)
    dy = np.abs(y_recon - y_target)
    # the reconstruction error is a position, so the tolerance has to scale
    # with the grid: a fixed number of degrees is a different demand at every
    # resolution, and on a coarse grid it fails on sound weights
    tol = 1e-4 * min(CFG["grid"]["dx"], CFG["grid"]["dy"])
    print(f"  position reconstruction: max |dlon| = {dx.max():.3e} deg, "
          f"max |dlat| = {dy.max():.3e} deg (tolerance {tol:.3e})")

    wsum = w0 + w1 + w2
    print(f"  weight sum:  max |1 - sum| = {np.abs(wsum - 1).max():.3e}")
    wmin = min(w0.min(), w1.min(), w2.min())
    wmax = max(w0.max(), w1.max(), w2.max())
    print(f"  weight range: [{wmin:.3e}, {wmax:.3e}]")

    ok = (dx.max() < tol) and (dy.max() < tol) and (wmin > -1e-5)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# check B -- speed invariance
# ---------------------------------------------------------------------------

def check_speed(wts, u_nat, v_nat, u_rot, v_rot, u_reg, v_reg):
    """|V| is invariant under a rotation of the frame -- but only POINTWISE.

    The comparison must therefore be made at the mesh nodes, before any
    interpolation.  Comparing an interpolated speed against the speed of the
    interpolated components does NOT work: the norm is convex, so
    interp(|V|) >= |interp(V)|, with a gap set by how much the velocity
    direction turns within one element.  Inside an eddy that gap is a sizeable
    fraction of the speed itself, so such a test fails on correct data.

    At nodes this catches non-orthogonal or mis-composed rotations.  It does
    NOT catch a wrong rotation angle -- every rotation preserves the norm.
    That is what check C is for.
    """
    hr("B. speed invariance under rotation (at mesh nodes)")

    spd_nat = np.sqrt(u_nat**2 + v_nat**2)
    spd_rot = np.sqrt(u_rot**2 + v_rot**2)

    good = np.isfinite(spd_nat) & np.isfinite(spd_rot)
    d = np.abs(spd_nat[good] - spd_rot[good])
    scale = spd_nat[good].max()
    print(f"  nodes compared: {int(good.sum())}")
    print(f"  max |d|speed|| = {d.max():.3e} m/s (max speed {scale:.3e} m/s)")
    print(f"  relative: {d.max() / scale:.3e}")

    ok = d.max() / scale < 1e-5

    # secondary, on the grid: convexity requires interp(|V|) >= |interp(V)|.
    # A violation beyond rounding means the two paths used different weights.
    spd_interp = wts.interpolate(spd_nat.astype(np.float32))
    spd_grid = np.sqrt(u_reg**2 + v_reg**2)
    both = np.isfinite(spd_interp) & np.isfinite(spd_grid)
    gap = spd_interp[both] - spd_grid[both]
    viol = gap.min()
    print(f"  convexity gap interp(|V|) - |interp(V)|: "
          f"min {viol:+.3e}, median {np.median(gap):+.3e} m/s")
    if viol < -1e-5 * scale:
        print("  !! convexity violated -- the two paths disagree on weights")
        ok = False

    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# check C -- vorticity cross-check
# ---------------------------------------------------------------------------

def zeta_on_mesh(lon, lat, elements, u_geo, v_geo):
    """Vertical vorticity per element, from geographic (east/north) velocities.

    Each element is projected orthographically onto the tangent plane at its
    own centroid, and the node velocity vectors are re-expressed in the
    centroid's east/north basis.  Doing it per element rather than in a global
    lon/lat frame is what keeps this well behaved at the pole, where the
    east/north basis rotates arbitrarily fast between neighbouring nodes.
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    clat, slat = np.cos(lat_r), np.sin(lat_r)
    clon, slon = np.cos(lon_r), np.sin(lon_r)

    p = np.stack([clat * clon, clat * slon, slat], axis=1)
    e_east = np.stack([-slon, clon, np.zeros_like(slon)], axis=1)
    e_north = np.stack([-slat * clon, -slat * slon, clat], axis=1)

    # node velocity as a 3D vector
    V3 = u_geo[:, None] * e_east + v_geo[:, None] * e_north

    idx = elements                      # (n_elem, 3)
    pe = p[idx]                         # (n_elem, 3, 3)

    c = pe.sum(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)

    # centroid tangent basis; fall back near the pole where z x c degenerates
    z = np.array([0.0, 0.0, 1.0])
    ee = np.cross(z, c)
    n_ee = np.linalg.norm(ee, axis=1, keepdims=True)
    degenerate = (n_ee[:, 0] < 1e-8)
    if degenerate.any():
        fallback = np.cross(np.array([1.0, 0.0, 0.0]), c[degenerate])
        ee[degenerate] = fallback
        n_ee[degenerate] = np.linalg.norm(fallback, axis=1, keepdims=True)
    ee /= n_ee
    en = np.cross(c, ee)

    # local planar coordinates and velocity components
    x = R_EARTH * np.einsum("eij,ej->ei", pe, ee)
    y = R_EARTH * np.einsum("eij,ej->ei", pe, en)
    U = np.einsum("eij,ej->ei", V3[idx], ee)
    V = np.einsum("eij,ej->ei", V3[idx], en)

    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
    twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    dVdx = ((y1 - y2) * V[:, 0] + (y2 - y0) * V[:, 1]
            + (y0 - y1) * V[:, 2]) / twoA
    dUdy = ((x2 - x1) * U[:, 0] + (x0 - x2) * U[:, 1]
            + (x1 - x0) * U[:, 2]) / twoA

    return dVdx - dUdy, np.abs(twoA) * 0.5


def elem_to_node(values, areas, elements, n_node):
    """Area-weighted average of an element field onto nodes, skipping
    elements whose value is not finite (i.e. touching dry nodes)."""
    good = np.isfinite(values)
    w = np.where(good, areas, 0.0)
    val = np.where(good, values, 0.0)

    num = np.zeros(n_node)
    den = np.zeros(n_node)
    np.add.at(num, elements.ravel(), np.repeat(val * w, 3))
    np.add.at(den, elements.ravel(), np.repeat(w, 3))

    out = np.full(n_node, np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def zeta_on_grid(u_reg, v_reg, lon_eq, lat_eq):
    """Spherical vorticity on the regular rotated grid.

    Arrays are (nx, ny) = (lon, lat), so lambda runs along axis 0 and phi
    along axis 1.
    """
    lam = np.radians(lon_eq.astype(np.float64))
    phi = np.radians(lat_eq.astype(np.float64))
    cosphi = np.cos(phi)[None, :]

    dv_dlam = np.gradient(v_reg, lam, axis=0)
    ducos_dphi = np.gradient(u_reg * cosphi, phi, axis=1)

    return (dv_dlam - ducos_dphi) / (R_EARTH * cosphi)


def check_vorticity(wts, lon, lat, elements, u_nat, v_nat, u_reg, v_reg):
    """Vertical vorticity is a scalar invariant under a sphere rotation, so
    computing it on the native mesh in geographic coordinates and then
    interpolating must agree with computing it from the gridded, rotated
    components.

    This is the one check that can fail even when the port is a faithful copy
    of the original code, because it tests the *physics* of the rotation chain
    rather than agreement between two implementations of it.
    """
    hr("C. vorticity cross-check on the rotation chain")

    n_node = lon.shape[0]

    def correlate(u_g, v_g, label):
        z_elem, area = zeta_on_mesh(lon, lat, elements, u_g, v_g)
        z_node = elem_to_node(z_elem, area, elements, n_node)
        z_native = wts.interpolate(z_node.astype(np.float32))

        z_grid = zeta_on_grid(u_reg.astype(np.float64),
                              v_reg.astype(np.float64),
                              wts.lon_eq, wts.lat_eq)

        both = np.isfinite(z_native) & np.isfinite(z_grid)
        a = z_native[both].astype(np.float64)
        b = z_grid[both]
        r = np.corrcoef(a, b)[0, 1]
        ratio = np.std(b) / np.std(a)
        print(f"  {label}")
        print(f"     correlation      = {r:+.4f}")
        print(f"     std(grid)/std(native) = {ratio:.4f}")
        print(f"     n points         = {both.sum()}")
        return r

    u_g, v_g = pf.vec_rotate_r2g(*MESH_ABG, lon, lat, u_nat, v_nat, flag=1)

    r_used = correlate(u_g, v_g,
                       "as configured (mesh-native -> geographic)")
    r_alt = correlate(u_nat, v_nat,
                      "control: velocities assumed already geographic")

    print()
    print(f"  implemented chain r = {r_used:+.4f}")
    print(f"  control chain     r = {r_alt:+.4f}")

    ok = (r_used > 0.9) and (r_used >= r_alt)
    if not ok and r_alt > r_used:
        print("  !! the control scores higher -- rotation.mesh_abg in "
              "config.yaml is probably wrong for this run")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("depth", type=int)
    ap.add_argument("--day", type=int, default=0,
                    help="index into the time axis (default 0)")
    ap.add_argument("--level-index", type=int, default=None,
                    help="force the nz1 index, bypassing the depth lookup")
    ap.add_argument("--weights", default=WEIGHTS_DIR)
    args = ap.parse_args()

    wts = Weights(args.weights)
    lon, lat, elements, nz1, max_dep, version = load_mesh_diag(MESH_PATH)

    lons_rot, lats_rot = pf.ut.scalar_g2r(ALPHA, BETA, GAMMA, lon, lat)

    if args.level_index is not None:
        iz = args.level_index
        print(f"  level {iz} at {nz1[iz]:.1f} m (forced), day index {args.day}")
    else:
        iz = int(np.argmin(np.abs(nz1 - args.depth)))
        print(f"  level {iz} at {nz1[iz]:.1f} m, day index {args.day}")

    ufile = os.path.join(DATA_PATH, f"unod.fesom.{args.year}.nc")
    vfile = os.path.join(DATA_PATH, f"vnod.fesom.{args.year}.nc")

    with xr.open_dataset(ufile) as ds:
        u_nat = ds.unod.isel(nz1=iz, time=args.day).values
        t_stamp = Timestamp(ds.time.values[args.day]).to_pydatetime()
    with xr.open_dataset(vfile) as ds:
        v_nat = ds.vnod.isel(nz1=iz, time=args.day).values

    u_nat = np.asarray(u_nat, dtype=np.float64)
    v_nat = np.asarray(v_nat, dtype=np.float64)

    print(f"  timestamp: {t_stamp}")

    # how is the sea floor represented?  Older FESOM writes dry nodes as 0.0,
    # which the interpolation would otherwise blend into every cell within one
    # element of the bathymetry.
    n_zero = int(((u_nat == 0.0) & (v_nat == 0.0)).sum())
    n_nan = int((~np.isfinite(u_nat)).sum())
    print(f"\n  sea floor: {n_nan} NaN nodes, {n_zero} exactly-zero nodes "
          f"of {u_nat.size}")
    dry = dry_mask(max_dep, nz1[iz], u_nat, v_nat, verbose=True)
    print(f"  masking {int(dry.sum())} dry nodes ({100 * dry.mean():.2f}%)")
    u_nat = np.where(dry, np.nan, u_nat)
    v_nat = np.where(dry, np.nan, v_nat)

    u_rot, v_rot = wts.rotate(u_nat.astype(np.float32),
                              v_nat.astype(np.float32))
    u_reg = wts.interpolate(u_rot)
    v_reg = wts.interpolate(v_rot)

    results = {}
    results["A weights"] = check_weights(wts, lons_rot, lats_rot)

    results["B speed"] = check_speed(
        wts,
        u_nat.astype(np.float32), v_nat.astype(np.float32),
        u_rot, v_rot, u_reg, v_reg,
    )
    results["C vorticity"] = check_vorticity(wts, lon, lat, elements,
                                             u_nat, v_nat, u_reg, v_reg)

    hr("summary")
    for k, v in results.items():
        print(f"  {k:20s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()