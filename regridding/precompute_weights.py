#!/usr/bin/env python
"""
One-off precomputation for the FESOM -> rotated regular grid interpolation.

Everything written here depends only on (mesh, target grid, Euler angles) --
NOT on year, day or depth.  Run this once on a node with ~96 GB, then every
production job just mmaps the results.

What gets written into <out_dir>:

    meta.json     grid definition, Euler angles, mesh path, node/element counts
    valid.npy     (nx, ny) bool   -- target cells inside mesh AND ocean AND region
    v0/v1/v2.npy  (nvalid,) int32 -- mesh node indices of the containing triangle
    w0/w1.npy     (nvalid,) f32   -- barycentric weights (w2 = 1 - w0 - w1)
    true_lon.npy  (nx, ny) f32    -- geographic longitude of each target cell
    true_lat.npy  (nx, ny) f32    -- geographic latitude  of each target cell
    rot_a/b/c/d.npy (nnod,) f64   -- combined vector-rotation coefficients

The 40 GB trifinder is built here, queried once for all 25M target points, and
then thrown away.  Its entire useful content is the triangle-index array, which
is 100 MB.

Usage:
    python precompute_weights.py [out_dir]
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import matplotlib.tri as mtri
import numpy as np
import pyfesom2 as pf

from config import euler_angles, load_config, meta_from_config
from mesh_diag import load_mesh_diag

# ----------------------------------------------------------------------------
# configuration -- everything comes from config.yaml
# ----------------------------------------------------------------------------

CFG = load_config()

MODE = CFG["mode"]
if MODE == "none":
    raise SystemExit(
        "mode is none: the input is already on a lon/lat grid, so there are "
        "no weights to build")

MESH_PATH = CFG["paths"]["mesh_path"]
OUT_DIR = CFG["paths"]["weights_dir"]

ALPHA, BETA, GAMMA = euler_angles(CFG)
MESH_ABG = CFG["rotation"]["mesh_abg"]

DX = CFG["grid"]["dx"]
DY = CFG["grid"]["dy"]
LEFT = CFG["grid"]["left"]
RIGHT = CFG["grid"]["right"]
BOTTOM = CFG["grid"]["bottom"]
TOP = CFG["grid"]["top"]


def build_target_grid():
    """Axes and meshed coordinates of the target grid, in the target frame.

    Rotating the pole onto the equator maps the polar cap onto the square of
    half-width (top - bottom), which is why the geographic bounds do not
    appear here; one extra cell along longitude keeps nx and ny unequal, so
    a transposed read fails loudly instead of silently.  Without rotation the
    target frame is the geographic one and the bounds are used as given.

    config.grid_shape derives the same sizes without building the arrays.
    """
    if MODE == "rotate_regrid":
        nx2 = TOP - BOTTOM + DX
        ny2 = TOP - BOTTOM
        lon_t = np.arange(-nx2, nx2, DX)
        lat_t = np.arange(-ny2, ny2, DY)
    else:
        lon_t = np.arange(LEFT, RIGHT, DX)
        lat_t = np.arange(BOTTOM, TOP, DY)

    xx, yy = np.meshgrid(lon_t, lat_t)
    return lon_t, lat_t, xx.T, yy.T  # -> (nx, ny)


def build_mask(xx_pol, yy_pol):
    """Land mask (True = masked out), cached in MESH_PATH.

    The mask is computed on the geographic coordinates of the target cells,
    so it belongs to one target grid and no other.  The cache name carries a
    digest of the settings that define that grid, so changing any of them
    builds a new mask instead of reusing one that no longer fits.
    """
    tag = hashlib.blake2b(
        json.dumps(meta_from_config(CFG), sort_keys=True).encode(),
        digest_size=4).hexdigest()
    cache = os.path.join(
        MESH_PATH,
        f"mask_ne_{MODE}_{xx_pol.shape[0]}x{xx_pol.shape[1]}_{tag}.npy")

    if os.path.exists(cache):
        print(f"  reusing cached Natural Earth mask {os.path.basename(cache)}")
        mask = np.load(cache)
    else:
        print(f"  building Natural Earth land mask (slow, one-off) "
              f"-> {os.path.basename(cache)}")
        mask = pf.ut.mask_ne(xx_pol, yy_pol)
        np.save(cache, mask)

    region = (
        (yy_pol >= BOTTOM) & (yy_pol <= TOP)
        & (xx_pol >= LEFT) & (xx_pol <= RIGHT)
    )
    mask = mask.copy()
    mask[~region] = True
    return mask


def rotation_coefficients(lons, lats):
    """Collapse the two-step vector rotation into four per-node coefficients.

        u_rot = a * u + b * v
        v_rot = c * u + d * v

    Both vec_rotate_* steps are linear and homogeneous in (u, v) -- they map
    the vector into 3D Cartesian, apply a rotation matrix, and project back --
    so the whole chain is a per-node 2x2 matrix.  Evaluating it on the unit
    vectors recovers that matrix, and the per-timestep cost drops from two
    trig-heavy rotations to four array multiplies.
    """
    n = lons.shape[0]

    def chain(u, v):
        u_g, v_g = pf.vec_rotate_r2g(*MESH_ABG, lons, lats, u, v, flag=1)
        return pf.vec_rotate_g2r(ALPHA, BETA, GAMMA, lons, lats, u_g, v_g, 1)

    ones = np.ones(n)
    zeros = np.zeros(n)

    a, c = chain(ones, zeros)   # response to u = 1, v = 0
    b, d = chain(zeros, ones)   # response to u = 0, v = 1

    # --- verify linearity on random input rather than assuming it ---
    rng = np.random.default_rng(0)
    ut, vt = rng.normal(size=n), rng.normal(size=n)
    u_ref, v_ref = chain(ut, vt)
    err = max(
        np.nanmax(np.abs(a * ut + b * vt - u_ref)),
        np.nanmax(np.abs(c * ut + d * vt - v_ref)),
    )
    print(f"  rotation linearity check: max abs error = {err:.3e}")
    if err > 1e-10:
        raise RuntimeError(
            "vector rotation is not linear to the expected precision -- "
            "do not use the precomputed coefficients"
        )

    # also confirm the map is homogeneous (no constant offset)
    u0, v0 = chain(zeros, zeros)
    if np.nanmax(np.abs(u0)) > 1e-12 or np.nanmax(np.abs(v0)) > 1e-12:
        raise RuntimeError("vector rotation has a nonzero constant term")

    return a, b, c, d


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR

    if os.path.exists(out_dir):
        print(f"{out_dir} already exists -- refusing to overwrite. "
              f"Delete it explicitly if you want to rebuild.")
        return

    print("loading mesh from fesom.mesh.diag.nc ...")
    model_lons, model_lats, elements, nz1, max_dep, version = load_mesh_diag(
        MESH_PATH)

    print(f"mapping mesh coordinates to the target frame ({MODE}) ...")
    lons_rot, lats_rot = pf.ut.scalar_g2r(
        ALPHA, BETA, GAMMA, model_lons, model_lats
    )

    # drop elements that wrap the target frame's dateline
    d = lons_rot[elements].max(axis=1) - lons_rot[elements].min(axis=1)
    no_cyclic_elem = np.argwhere(d < 100).ravel()
    tris = elements[no_cyclic_elem]
    print(f"  {tris.shape[0]} of {elements.shape[0]} elements kept")

    print("building target grid ...")
    lon_eq, lat_eq, xx_eq, yy_eq = build_target_grid()
    nx, ny = xx_eq.shape
    print(f"  target grid: {nx} x {ny} = {nx * ny / 1e6:.1f}M points")

    # geographic coordinates of every target cell -- needed downstream for f,
    # for the CE/AE polarity rule, and for writing real lon/lat to file.
    # Without rotation this is the identity and returns the grid itself.
    print("computing true (geographic) coordinates of target cells ...")
    true_lon, true_lat = pf.ut.scalar_r2g(ALPHA, BETA, GAMMA, xx_eq, yy_eq)

    print("building land/region mask ...")
    mask = build_mask(true_lon, true_lat)

    print("building trifinder (slow, ~15 min, high memory) ...")
    triang = mtri.Triangulation(lons_rot, lats_rot, tris)
    trifinder = triang.get_trifinder()

    print("querying trifinder for all target points ...")
    tri_idx = trifinder(xx_eq, yy_eq)

    # the 40 GB object has now told us everything it knows
    del trifinder, triang

    valid = (tri_idx >= 0) & (~mask)
    nvalid = int(valid.sum())
    print(f"  {nvalid / 1e6:.1f}M valid target cells "
          f"({100 * nvalid / (nx * ny):.1f}%)")

    print("computing barycentric weights ...")
    ti = tri_idx[valid]
    v0, v1, v2 = tris[ti].T.astype(np.int32)

    x = xx_eq[valid]
    y = yy_eq[valid]
    x0, y0 = lons_rot[v0], lats_rot[v0]
    x1, y1 = lons_rot[v1], lats_rot[v1]
    x2, y2 = lons_rot[v2], lats_rot[v2]

    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
    w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
    w2 = 1.0 - w0 - w1

    # sanity: every point should sit inside its triangle
    tol = 1e-6
    bad = (w0 < -tol) | (w1 < -tol) | (w2 < -tol)
    print(f"  points outside their own triangle: {int(bad.sum())}")
    if bad.sum() > 0.001 * nvalid:
        raise RuntimeError("too many points fail the barycentric containment "
                           "test -- triangle indexing is probably wrong")

    print("computing vector-rotation coefficients ...")
    rot_a, rot_b, rot_c, rot_d = rotation_coefficients(model_lons, model_lats)

    # the settings the weights depend on, taken from the same helper the
    # loader compares against -- so the two can never drift apart
    meta = dict(meta_from_config(CFG))
    meta.update({
        "mesh_path": MESH_PATH,
        "diag_layout": version,
        "nx": int(nx), "ny": int(ny),
        "n_nodes": int(model_lons.shape[0]),
        "n_valid": nvalid,
        "coordinate_frame": (
            "rotated: north pole moved onto the equator"
            if MODE == "rotate_regrid" else "geographic"),
        "built_from_config": CFG["_path"],
    })

    print(f"writing to {out_dir} ...")
    tmp_dir = tempfile.mkdtemp(
        dir=os.path.dirname(os.path.abspath(out_dir)) or ".",
        prefix=".regrid_weights_tmp_",
    )
    try:
        def save(name, arr):
            np.save(os.path.join(tmp_dir, name), arr, allow_pickle=False)

        save("valid.npy", valid)
        save("v0.npy", v0)
        save("v1.npy", v1)
        save("v2.npy", v2)
        save("w0.npy", w0.astype(np.float32))
        save("w1.npy", w1.astype(np.float32))
        save("lon_eq.npy", lon_eq.astype(np.float32))
        save("lat_eq.npy", lat_eq.astype(np.float32))
        save("true_lon.npy", true_lon.astype(np.float32))
        save("true_lat.npy", true_lat.astype(np.float32))
        save("rot_a.npy", rot_a)
        save("rot_b.npy", rot_b)
        save("rot_c.npy", rot_c)
        save("rot_d.npy", rot_d)

        with open(os.path.join(tmp_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        os.chmod(tmp_dir, 0o755)
        os.rename(tmp_dir, out_dir)  # atomic: dest did not exist
        tmp_dir = None
    finally:
        if tmp_dir is not None and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    print("done.")
    print(f"  true latitude range in domain: "
          f"{np.nanmin(true_lat[valid]):.2f} .. {np.nanmax(true_lat[valid]):.2f}")


if __name__ == "__main__":
    main()