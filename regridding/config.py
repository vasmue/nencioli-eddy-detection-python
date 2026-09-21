"""
Loader for config.yaml.

Beyond reading the file, this module provides the consistency check that makes
the shared config actually safer than duplicated constants: the mode, rotation
and grid settings are stamped into the weights directory when it is built, and
`check_meta_matches` compares them on every load.  Editing config.yaml after
building weights therefore raises instead of silently producing output on a
grid that no longer matches the weights.

Which sections are needed depends on `mode`; see MODES below and the comments
in config.yaml.

The config is config.yaml next to this module, unless a path is passed in.
"""

import os

import numpy as np
import yaml

MODES = ("rotate_regrid", "regrid", "none")


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config.yaml")
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    _validate(cfg, path)
    cfg["_path"] = path
    return cfg


def _validate(cfg, path):
    if "mode" not in cfg:
        raise ValueError(f"{path}: missing 'mode'; one of {', '.join(MODES)}")
    mode = cfg["mode"]
    if mode not in MODES:
        raise ValueError(f"{path}: mode is {mode!r}, must be one of "
                         f"{', '.join(MODES)}")

    if mode == "none":
        return

    required = {
        "paths": ["mesh_path", "data_path", "out_root", "weights_dir"],
        "rotation": (["alpha", "beta", "gamma", "mesh_abg"]
                     if mode == "rotate_regrid" else ["mesh_abg"]),
        "grid": ["dx", "dy", "left", "right", "bottom", "top"],
        "output": ["time_units", "calendar", "overwrite", "min_valid_mb",
                   "complevel", "chunk_lon", "chunk_lat"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"{path}: mode {mode} needs section '{section}'")
        for k in keys:
            if k not in cfg[section]:
                raise ValueError(f"{path}: mode {mode} needs "
                                 f"'{section}.{k}'")

    abg = cfg["rotation"]["mesh_abg"]
    if abg is None or len(abg) != 3:
        raise ValueError(f"{path}: rotation.mesh_abg must be 3 angles; "
                         f"[0, 0, 0] is the identity")

    g = cfg["grid"]
    if g["dx"] <= 0 or g["dy"] <= 0:
        raise ValueError(f"{path}: grid.dx and grid.dy must be positive")
    if g["left"] >= g["right"]:
        raise ValueError(f"{path}: grid.left must be west of grid.right")
    if g["bottom"] >= g["top"]:
        raise ValueError(f"{path}: grid.bottom must be below grid.top")
    if not (-180 <= g["left"] and g["right"] <= 180):
        raise ValueError(f"{path}: grid.left and grid.right must lie within "
                         f"-180 .. 180")
    if not (-90 <= g["bottom"] and g["top"] <= 90):
        raise ValueError(f"{path}: grid.bottom and grid.top must lie within "
                         f"-90 .. 90")

    if mode == "rotate_regrid" and not (g["left"] == -180 and g["right"] == 180
                                        and g["top"] == 90):
        raise ValueError(
            f"{path}: mode rotate_regrid needs the whole polar cap "
            f"(left -180, right 180, top 90), and has left {g['left']}, "
            f"right {g['right']}, top {g['top']}.\n"
            f"The rotated target grid is the square of half-width "
            f"(top - bottom) that the cap maps onto, which describes the "
            f"domain only when the domain is the cap itself. Use mode "
            f"regrid for anything smaller."
        )


def grid_shape(cfg):
    """Number of target cells along longitude and latitude.

    rotate_regrid: the cap maps onto a square of half-width (top - bottom)
    in the rotated frame, widened by one cell along longitude so that nx and
    ny differ and a transposed read fails loudly.
    regrid: the geographic domain as given.
    """
    mode = cfg["mode"]
    if mode == "none":
        raise ValueError("mode none defines no target grid")

    g = cfg["grid"]
    if mode == "rotate_regrid":
        nx2 = g["top"] - g["bottom"] + g["dx"]
        ny2 = g["top"] - g["bottom"]
        lon = np.arange(-nx2, nx2, g["dx"])
        lat = np.arange(-ny2, ny2, g["dy"])
    else:
        lon = np.arange(g["left"], g["right"], g["dx"])
        lat = np.arange(g["bottom"], g["top"], g["dy"])
    return lon.size, lat.size


def euler_angles(cfg):
    """Geographic -> target frame, as three Euler angles.

    regrid leaves the frame alone, which is the identity rotation, so it has
    no angles of its own.
    """
    mode = cfg["mode"]
    if mode == "none":
        raise ValueError("mode none defines no target frame")
    if mode != "rotate_regrid":
        return [0, 0, 0]
    r = cfg["rotation"]
    return [r["alpha"], r["beta"], r["gamma"]]


def meta_from_config(cfg):
    """The subset of the config that the weights depend on.  Anything in here
    invalidates an existing weights directory if changed."""
    mode = cfg["mode"]
    if mode == "none":
        raise ValueError("mode none builds no weights")

    r, g = cfg["rotation"], cfg["grid"]
    euler = euler_angles(cfg)
    return {
        "mode": mode,
        "euler_alpha": euler[0],
        "euler_beta": euler[1],
        "euler_gamma": euler[2],
        "mesh_abg": list(r["mesh_abg"]),
        "dx": g["dx"], "dy": g["dy"],
        "left": g["left"], "right": g["right"],
        "bottom": g["bottom"], "top": g["top"],
    }


def check_meta_matches(cfg, meta, weights_dir):
    """Raise if a weights directory was built with different settings."""
    want = meta_from_config(cfg)
    bad = []
    for k, v in want.items():
        if k not in meta:
            bad.append(f"  {k}: missing from meta.json (weights predate it?)")
        elif meta[k] != v:
            bad.append(f"  {k}: weights have {meta[k]!r}, config says {v!r}")

    if bad:
        raise ValueError(
            f"config.yaml does not match the weights in {weights_dir}:\n"
            + "\n".join(bad)
            + "\n\nEither restore the settings the weights were built with, "
              "or rebuild the weights with precompute_weights.py."
        )