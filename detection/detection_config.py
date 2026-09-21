"""
Loader for detection.yaml.

Separate from the regridding's config.py: the detection reads gridded u/v from
netCDF and shares nothing with the FESOM pipeline, so it should not inherit
its mesh paths, Euler angles or weights-consistency checks.

The config is detection.yaml next to this module, unless a path is passed in.
"""

import os

import yaml


def load_detection_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "detection.yaml")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"detection config not found: {path}")

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    if "detection" not in cfg:
        raise ValueError(f"{path}: missing section 'detection'")
    d = cfg["detection"]

    for k in ("input", "a", "b", "centers_out"):
        if k not in d:
            raise ValueError(f"{path}: missing 'detection.{k}'")
    for k in ("u_var", "v_var", "lon_var", "lat_var", "time_var"):
        if k not in d["input"]:
            raise ValueError(f"{path}: missing 'detection.input.{k}'")

    d.setdefault("regions", [])
    d.setdefault("rotated", False)
    if d["rotated"] and "euler" not in d:
        raise ValueError(f"{path}: detection.rotated is true but "
                         f"'detection.euler' is missing")
    if d["a"] < 1 or d["b"] < 1:
        raise ValueError(f"{path}: detection.a and detection.b must be >= 1")

    if "shape" in cfg:
        s = cfg["shape"]
        for k in ("shapes_out", "dz"):
            if k not in s:
                raise ValueError(f"{path}: missing 'shape.{k}'")
        s.setdefault("fac_step", 0.5)
        s.setdefault("fac_max", 3.0)
        s.setdefault("min_contours", 2)
        s.setdefault("contour_points", 50)
        if s["dz"] <= 0:
            raise ValueError(f"{path}: shape.dz must be > 0")
        if s["fac_step"] <= 0 or s["fac_max"] <= 1:
            raise ValueError(f"{path}: shape.fac_step must be > 0 "
                             f"and shape.fac_max > 1")

    cfg.setdefault("output", {})
    cfg["output"].setdefault("time_units", "hours since 1990-01-01 00:00:00")
    cfg["output"].setdefault("calendar", "standard")

    cfg["_path"] = path
    return cfg
