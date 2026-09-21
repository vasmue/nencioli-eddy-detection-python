#!/usr/bin/env python
"""
Plot one day: speed as colour, velocity as arrows, and the effective contour
of every eddy -- red for anticyclones, blue for cyclones, dashed where the
contour is only the largest closed one.

Usage:
    python plot_day.py UV_FILE SHAPES_FILE [--out plot.png]
                       [--lon MIN MAX] [--lat MIN MAX] [--max-err 70]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from detection_config import load_detection_config
from run_center_detection import read_coords, read_uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uv_file")
    ap.add_argument("shapes_file")
    ap.add_argument("--out", default="plot.png")
    ap.add_argument("--lon", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="longitude range to plot; default the whole file")
    ap.add_argument("--lat", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="latitude range to plot; default the whole file")
    ap.add_argument("--max-err", type=float, default=70.0,
                    help="plot only eddies with a shape error below this, "
                         "in percent (default 70)")
    args = ap.parse_args()

    inp = load_detection_config()["detection"]["input"]
    u, v, date = read_uv(args.uv_file, inp)
    lon, lat = read_coords(args.uv_file, inp)

    lon_rng = args.lon or (lon.min(), lon.max())
    lat_rng = args.lat or (lat.min(), lat.max())
    cols = (lon[0] >= lon_rng[0]) & (lon[0] <= lon_rng[1])
    rows = (lat[:, 0] >= lat_rng[0]) & (lat[:, 0] <= lat_rng[1])
    lon, lat = lon[rows][:, cols], lat[rows][:, cols]
    u, v = u[rows][:, cols], v[rows][:, cols]
    speed = np.hypot(u, v)

    fig, ax = plt.subplots(figsize=(14, 8))

    pc = ax.pcolormesh(lon, lat, speed, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax, label="speed (m/s)")

    # about 100 arrows across the plotted region; an arrow at the 95th
    # percentile speed is as long as the spacing between arrows
    s = max(1, u.shape[1] // 100)
    step = s * abs(lon[0, 1] - lon[0, 0])
    ax.quiver(lon[::s, ::s], lat[::s, ::s], u[::s, ::s], v[::s, ::s],
              color="k", angles="xy", scale_units="xy",
              scale=np.nanpercentile(speed, 95) / step, width=0.0008)

    with xr.open_dataset(args.shapes_file, decode_times=False) as ds:
        clon = ds["effective_contour_lon"].values
        clat = ds["effective_contour_lat"].values
        kind = ds["type"].values
        cen_lon = ds["lon"].values
        cen_lat = ds["lat"].values
        weak = ds["weak_shape"].values == 1
        # nested centres are secondary minima inside another eddy
        good = ((ds["shape_error"].values < args.max_err) &
                (ds["nested"].values == 0))

    clon, clat, kind = clon[good], clat[good], kind[good]
    cen_lon, cen_lat, weak = cen_lon[good], cen_lat[good], weak[good]

    # dashed: weak shape -- no contour passed the speed test, so the shape
    # is only the largest closed contour
    for x, y, k, lg in zip(clon, clat, kind, weak):
        ax.plot(x, y, color="blue" if k == 1 else "red", lw=1,
                ls="--" if lg else "-")

    # count only the eddies whose centre lies in the plotted region
    inside = ((cen_lon >= lon_rng[0]) & (cen_lon <= lon_rng[1]) &
              (cen_lat >= lat_rng[0]) & (cen_lat <= lat_rng[1]))
    kind = kind[inside]

    ax.set_xlim(lon_rng)
    ax.set_ylim(lat_rng)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{date:%Y-%m-%d}: {int((kind == 1).sum())} cyclones (blue), "
                 f"{int((kind == -1).sum())} anticyclones (red)")
    fig.savefig(args.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()