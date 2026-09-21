"""
Reader for fesom.mesh.diag.nc, handling both the old and new FESOM2 layouts.

Replaces pyfesom2.load_mesh for the purposes of these scripts -- which also
means no mesh pickle gets written, so concurrent jobs have nothing to race on.

Layout differences handled here:
    old:  elements   (1-based, shape (3, nelem)),  nz1 stored NEGATIVE
    new:  face_nodes (1-based, shape (3, nelem)),  nz1 stored POSITIVE

Node coordinates are degrees in both layouts.
"""

import os

import numpy as np
import xarray as xr


def load_mesh_diag(mesh_path, verbose=True):
    """Return (lons, lats, elements, nz1, max_dep, version).

    lons, lats : float64 (n_node,) in degrees
    elements   : int32 (n_elem, 3), 0-based node indices
    nz1        : float64 (n_lev,) layer mid-depths, POSITIVE down
    max_dep    : float64 (n_node,) sea-floor depth, POSITIVE down, or None
    version    : 'old' or 'new'
    """
    path = os.path.join(mesh_path, "fesom.mesh.diag.nc")
    with xr.open_dataset(path) as diag:
        lon = np.asarray(diag.lon.values, dtype=np.float64)
        lat = np.asarray(diag.lat.values, dtype=np.float64)

        if "elements" in diag:
            version = "old"
            elements = np.asarray(diag.elements.values) - 1
            nz1 = -np.asarray(diag.nz1.values, dtype=np.float64)  # stored neg.
        else:
            version = "new"
            elements = np.asarray(diag.face_nodes.values) - 1
            nz1 = np.asarray(diag.nz1.values, dtype=np.float64)   # stored pos.

        # Bottom depth per node.  Sign convention varies between builds, so
        # normalise: if nothing is negative it is already positive-down,
        # otherwise flip it.
        if "zbar_n_bottom" in diag:
            zb = np.asarray(diag.zbar_n_bottom.values, dtype=np.float64)
            max_dep = zb if (zb < 0).sum() == 0 else -zb
        else:
            max_dep = None

    # stored as (3, nelem); we want (nelem, 3)
    if elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T
    elements = np.ascontiguousarray(elements, dtype=np.int32)

    n_node = lon.shape[0]
    if elements.min() < 0 or elements.max() >= n_node:
        raise ValueError(
            f"element indices out of range after 1-based correction: "
            f"[{elements.min()}, {elements.max()}] for {n_node} nodes"
        )

    if verbose:
        print(f"  fesom.mesh.diag.nc: '{version}' layout")
        print(f"  {n_node} nodes, {elements.shape[0]} elements, "
              f"{nz1.shape[0]} levels")
        print(f"  nz1 range: {nz1.min():.1f} .. {nz1.max():.1f} m (positive down)")
        if max_dep is None:
            print("  no zbar_n_bottom in diag file -- dry nodes cannot be "
                  "identified from bathymetry")
        else:
            print(f"  bottom depth: {max_dep.min():.1f} .. "
                  f"{max_dep.max():.1f} m (positive down)")

    return lon, lat, elements, nz1, max_dep, version
