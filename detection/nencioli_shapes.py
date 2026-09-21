"""
Eddy shape stage: windowed streamfunction from the velocity field.

Solves the Poisson problem

    div(grad psi) = zeta

on a box around each detected centre, with psi related to velocity by
u = -d(psi)/dy and v = d(psi)/dx.

DISCRETISATION
    Finite volume on the lon/lat cells.  Each cell equation balances the
    fluxes of grad(psi) through its four faces.  Only faces between two
    ocean cells carry an equation: at every other face -- window edge or
    coast -- the natural (Neumann) condition applies, with d(psi)/dn given
    by the tangential velocity there.  Written this way the discrete
    compatibility condition holds identically, because each interior face
    appears twice with opposite sign, so no correction of the right-hand
    side is needed.

    The resulting system is the weighted least-squares fit of grad(psi) to
    the velocity field rotated by ninety degrees, which is the best that can
    be asked of a flow that is not exactly non-divergent.

CACHED OPERATORS
    The operator depends on the window's shape, its ocean mask and the
    latitudes of its rows -- not on where the window sits in longitude, and
    not on the velocities.  Latitudes are quantised so that cos(lat) varies
    by at most COS_TOL within a bin, which makes the key discrete and the
    factorisation reusable across centres and across days.

ARRAY ORIENTATION
    Windows are (lat, lon) as in the centre stage, with latitude ascending
    along the first axis.

GRID LATITUDE vs TRUE LATITUDE
    The metric terms use the latitude of the grid the data is stored on: for
    a rotated grid that is the rotated latitude, which is what sets the
    spacing.  True latitude enters only through the Coriolis parameter when
    psi is converted to an equivalent surface elevation.

CONTOURS
    psi is turned into an equivalent surface elevation and contoured at a
    fixed step dz in metres, so levels mean the same thing for every eddy
    and on every day.  The effective contour is the largest closed contour
    around the centre across which speed increases outward at its northern,
    southern, eastern and western extremes; if no contour passes, it is the
    largest closed contour, and the eddy is flagged.  The speed contour is
    the closed contour with the highest mean speed along it.  The window is
    grown while the effective contour reaches its edge.
"""

import numpy as np
from contourpy import contour_generator
from scipy.ndimage import label, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

R_EARTH = 6371000.0
OMEGA = 7.2921159e-5
G = 9.81

# Widest relative spread of cos(lat) tolerated inside one operator cache bin.
COS_TOL = 0.005

# Beyond this grid latitude the east-west spacing collapses and the operator
# becomes badly conditioned.  Rotate the domain instead.
MAX_GRID_LAT = 70.0

_OPERATORS = {}


def _lat_bin(lat):
    """Bin index for a grid latitude, uniform in log(cos)."""
    return int(round(np.log(np.cos(np.radians(lat))) / COS_TOL))


def _bin_lat(k, lat):
    """Representative latitude of bin k, carrying the sign of lat."""
    rep = np.degrees(np.arccos(np.exp(k * COS_TOL)))
    return -rep if lat < 0 else rep


def _metrics(lat_rows, dlon, dlat):
    """East-west spacing per row, north-south spacing, and the east-west
    face length at the row interfaces.  All in metres."""
    dl = np.radians(abs(dlon))
    dp = np.radians(abs(dlat))
    dx = R_EARTH * np.cos(np.radians(lat_rows)) * dl
    dy = R_EARTH * dp
    lat_face = 0.5 * (lat_rows[:-1] + lat_rows[1:])
    dx_face = R_EARTH * np.cos(np.radians(lat_face)) * dl
    return dx, dy, dx_face


def _build_operator(ocean, lat_rows, dlon, dlat, pin):
    """Factorised flux operator for one window shape and ocean mask.

    Returns the LU factorisation and the map from grid cell to unknown.
    """
    dx, dy, dx_face = _metrics(lat_rows, dlon, dlat)

    idx = np.full(ocean.shape, -1, dtype=np.int64)
    n = int(ocean.sum())
    idx[ocean] = np.arange(n)

    # faces between horizontally adjacent ocean cells
    fi, fj = np.nonzero(ocean[:, :-1] & ocean[:, 1:])
    a1, a2 = idx[fi, fj], idx[fi, fj + 1]
    w_ew = dy / dx[fi]

    # faces between vertically adjacent ocean cells
    gi, gj = np.nonzero(ocean[:-1, :] & ocean[1:, :])
    b1, b2 = idx[gi, gj], idx[gi + 1, gj]
    w_ns = dx_face[gi] / dy

    rows = np.concatenate([a1, a1, a2, a2, b1, b1, b2, b2])
    cols = np.concatenate([a1, a2, a2, a1, b1, b2, b2, b1])
    vals = np.concatenate([-w_ew, w_ew, -w_ew, w_ew,
                           -w_ns, w_ns, -w_ns, w_ns])

    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tolil()

    # psi is defined only up to a constant; fixing one cell makes the
    # operator non-singular and so factorisable
    A[pin, :] = 0
    A[pin, pin] = 1.0

    return splu(A.tocsc()), idx


def _operator(ocean, lat_rows, dlon, dlat, pin):
    key = (ocean.shape, _lat_bin(lat_rows[0]), pin, ocean.tobytes())
    hit = _OPERATORS.get(key)
    if hit is None:
        hit = _build_operator(ocean, lat_rows, dlon, dlat, pin)
        _OPERATORS[key] = hit
    return hit


def window_ocean(u, v, ci, cj):
    """Ocean cells of the window that are connected to the centre.

    Pieces cut off by land carry their own additive constant and would make
    the operator singular, so only the piece holding the centre is solved.
    """
    ocean = np.isfinite(u) & np.isfinite(v)
    lab, _ = label(ocean)
    return lab == lab[ci, cj]


def streamfunction(u, v, ci, cj, lat_center, dlon, dlat):
    """Streamfunction over one window, in m^2/s.

    u, v are the window velocities in (lat, lon) with land as NaN, ci and cj
    the centre's indices within the window, lat_center its grid latitude,
    and dlon, dlat the grid spacing in degrees.

    Returns psi with NaN on land and on any ocean not connected to the
    centre, and the ocean mask that was solved on.
    """
    if abs(lat_center) > MAX_GRID_LAT:
        raise ValueError(
            f"grid latitude {lat_center:.2f} exceeds {MAX_GRID_LAT}; "
            f"rotate the domain")

    ocean = window_ocean(u, v, ci, cj)

    # latitudes of the window rows, taken from the quantised centre latitude
    # so that the operator depends on the bin and not on the exact position
    lat0 = _bin_lat(_lat_bin(lat_center), lat_center)
    lat_rows = lat0 + (np.arange(ocean.shape[0]) - ci) * abs(dlat)

    dx, dy, dx_face = _metrics(lat_rows, dlon, dlat)

    pin_cell = int(np.count_nonzero(ocean[:ci, :]) +
                   np.count_nonzero(ocean[ci, :cj]))
    lu, idx = _operator(ocean, lat_rows, dlon, dlat, pin_cell)

    rhs = np.zeros(int(ocean.sum()))

    # east-west faces: outward normal is +x, so d(psi)/dn is v
    fi, fj = np.nonzero(ocean[:, :-1] & ocean[:, 1:])
    flux = 0.5 * (v[fi, fj] + v[fi, fj + 1]) * dy
    np.add.at(rhs, idx[fi, fj], flux)
    np.add.at(rhs, idx[fi, fj + 1], -flux)

    # north-south faces: outward normal is +y, so d(psi)/dn is -u
    gi, gj = np.nonzero(ocean[:-1, :] & ocean[1:, :])
    flux = -0.5 * (u[gi, gj] + u[gi + 1, gj]) * dx_face[gi]
    np.add.at(rhs, idx[gi, gj], flux)
    np.add.at(rhs, idx[gi + 1, gj], -flux)

    rhs[pin_cell] = 0.0

    psi = np.full(ocean.shape, np.nan)
    psi[ocean] = lu.solve(rhs)
    return psi, ocean


def remove_ramp(psi, ocean, ci, cj, lat_center, dlon, dlat):
    """Subtract the best-fit plane in psi over the window.

    The plane is the local background flow; what is left is the circulation
    belonging to the eddy.
    """
    lat_rows = lat_center + (np.arange(psi.shape[0]) - ci) * abs(dlat)
    dx, dy, _ = _metrics(lat_rows, dlon, dlat)

    jj, ii = np.meshgrid(np.arange(psi.shape[1]) - cj,
                         np.arange(psi.shape[0]) - ci)
    x = jj * dx[:, None]
    y = ii * dy

    M = np.column_stack([np.ones(ocean.sum()), x[ocean], y[ocean]])
    coef, *_ = np.linalg.lstsq(M, psi[ocean], rcond=None)

    return psi - (coef[0] + coef[1] * x + coef[2] * y)


def equivalent_elevation(psi, true_lat):
    """Surface elevation equivalent to psi, in metres, at true latitude."""
    f = 2.0 * OMEGA * np.sin(np.radians(true_lat))
    return f * psi / G


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------

def _window_metres(row, col, ci, cj, lat_center, dlon, dlat):
    """Positions in metres relative to the centre, for fractional window
    row and column indices."""
    lat = lat_center + (row - ci) * abs(dlat)
    x = (col - cj) * R_EARTH * np.cos(np.radians(lat)) * np.radians(abs(dlon))
    y = (row - ci) * R_EARTH * np.radians(abs(dlat))
    return x, y


def _encloses(x, y, px, py):
    """Crossing-number test for a point inside a closed polygon."""
    x1, y1 = x[:-1], y[:-1]
    x2, y2 = x[1:], y[1:]
    s = (y1 > py) != (y2 > py)
    x1, y1, x2, y2 = x1[s], y1[s], x2[s], y2[s]
    xc = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    return int(np.count_nonzero(xc > px)) % 2 == 1


def _area(x, y):
    """Polygon area from the shoelace formula."""
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _shape_error(x, y, radius):
    """Area between the contour and the circle of equal area, as a
    percentage of the contour area.

    Both are measured as a function of azimuth about the centre, which is
    exact for a contour that is star-shaped about it.
    """
    r = np.hypot(x, y)
    th = np.unwrap(np.arctan2(y, x))
    dth = np.abs(np.diff(th))
    mid = 0.5 * (r[:-1] + r[1:])
    xor = 0.5 * np.sum(np.abs(mid ** 2 - radius ** 2) * dth)
    return 100.0 * xor / (np.pi * radius ** 2)


def _resample(x, y, n):
    """Even spacing of n points along a closed ring."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    t = np.linspace(0.0, s[-1], n)
    return np.interp(t, s, x), np.interp(t, s, y)


def _rings(eta, ci, cj, dz):
    """Closed contours of eta that enclose the centre, with their levels."""
    finite = np.isfinite(eta)
    if not finite.any():
        return []

    eta_c = eta[ci, cj]
    k0 = int(np.ceil((np.nanmin(eta) - eta_c) / dz))
    k1 = int(np.floor((np.nanmax(eta) - eta_c) / dz))
    if k1 < k0:
        return []

    gen = contour_generator(z=eta, line_type="Separate")

    out = []
    for k in range(k0, k1 + 1):
        level = eta_c + k * dz
        for line in gen.lines(level):
            x, y = line[:, 0], line[:, 1]
            if x[0] != x[-1] or y[0] != y[-1] or x.size < 4:
                continue
            if _encloses(x, y, cj, ci):
                out.append((level, y, x))          # level, rows, columns
    return out


def _speed_increases(row, col, vel):
    """Speed is higher just outside the contour than just inside it, at each
    of its northern, southern, eastern and western extremes.

    Where several vertices share an extreme, the one furthest east is taken
    for north, west for south, south for east and north for west.  Speed is
    sampled 0.05 grid cells either side of the contour.
    """
    d = 0.05

    rn = row.max()
    cn = col[row == rn].max()
    rs = row.min()
    cs = col[row == rs].min()
    ce = col.max()
    re = row[col == ce].min()
    cw = col.min()
    rw = row[col == cw].max()

    # inside and outside sample points at each extreme
    rows = np.array([rn - d, rn + d, rs + d, rs - d, re, re, rw, rw])
    cols = np.array([cn, cn, cs, cs, ce - d, ce + d, cw + d, cw - d])
    s = map_coordinates(vel, [rows, cols], order=1)
    return bool(np.all(s[0::2] <= s[1::2]))


def _shape(eta, vel, ci, cj, lat_center, dlon, dlat, dz, npts, min_rings):
    """Effective and speed contour of one window.

    Returns None unless at least min_rings closed contours enclose the
    centre: fewer than that and the shape is set by the contour step rather
    than measured from the field.
    """
    rings = _rings(eta, ci, cj, dz)
    if len(rings) < min_rings:
        return None

    areas, speeds = [], []
    for _, row, col in rings:
        x, y = _window_metres(row, col, ci, cj, lat_center, dlon, dlat)
        areas.append(_area(x, y))
        speeds.append(np.nanmean(
            map_coordinates(np.nan_to_num(vel), [row, col], order=1)))

    spd = int(np.argmax(speeds))

    vel0 = np.nan_to_num(vel)
    eff, large = int(np.argmax(areas)), True
    for k in np.argsort(areas)[::-1]:
        if _speed_increases(rings[k][1], rings[k][2], vel0):
            eff, large = int(k), False
            break

    nlat, nlon = eta.shape
    row, col = rings[eff][1], rings[eff][2]
    touches = bool(row.min() < 1 or col.min() < 1 or
                   row.max() > nlat - 2 or col.max() > nlon - 2)

    r_eff = np.sqrt(areas[eff] / np.pi)
    x, y = _window_metres(row, col, ci, cj, lat_center, dlon, dlat)

    out = {
        "amplitude": abs(rings[eff][0] - eta[ci, cj]),
        "effective_radius": r_eff,
        "speed_radius": np.sqrt(areas[spd] / np.pi),
        "speed": speeds[spd],
        "shape_error": _shape_error(x, y, r_eff),
        "n_contours": len(rings),
        "large": large,
        "touches": touches,
    }
    for name, k in (("effective", eff), ("speed", spd)):
        r, c = _resample(rings[k][1], rings[k][2], npts)
        out[name + "_row"] = r
        out[name + "_col"] = c
    return out


def nested(shapes, i, j, kind):
    """Flag eddies whose centre lies inside the effective contour of a
    larger eddy of the same type.

    Each eddy's contour comes from its own window, so two centres found in
    one eddy can each produce a ring, one inside the other.  The outer ring
    describes the eddy; the inner centre is a secondary speed minimum within
    it.  shapes are the results of eddy_shape, and i, j, kind the grid
    indices and type of the same eddies.
    """
    n = len(shapes)
    rows = [np.asarray(s["effective_row"]) for s in shapes]
    cols = [np.asarray(s["effective_col"]) for s in shapes]
    box = np.array([(r.min(), r.max(), c.min(), c.max())
                    for r, c in zip(rows, cols)]).reshape(n, 4)
    radius = np.array([s["effective_radius"] for s in shapes])

    flag = np.zeros(n, dtype=bool)
    for a in range(n):
        cand = np.flatnonzero((box[:, 0] <= i[a]) & (box[:, 1] >= i[a]) &
                              (box[:, 2] <= j[a]) & (box[:, 3] >= j[a]) &
                              (kind == kind[a]) & (radius >= radius[a]))
        for b in cand:
            if b != a and _encloses(cols[b], rows[b], j[a], i[a]):
                flag[a] = True
                break
    return flag


def eddy_shape(u, v, i, j, lat_center, true_lat, dlon, dlat,
               radii, dz, npts, min_rings):
    """Shape of the eddy centred on grid cell (i, j).

    u and v are the whole day's fields in (lat, lon), lat_center and
    true_lat the centre's grid and geographic latitude, radii the window
    half-widths to try in order, dz the contour step in metres, npts the
    number of vertices each contour is resampled to and min_rings the
    number of closed contours a window must produce to be used.

    A window that produces too few contours is not a rejection of the
    centre, only of that window size, so the search moves to the next one.
    The search stops at the first window whose effective contour stays clear
    of the edge, or whose effective contour is only the largest closed one.
    A larger window whose contours all fail the speed test does not replace
    the result from the smaller one.

    Contour vertices are returned as fractional indices into the whole grid.
    Returns None if no window qualifies.
    """
    nlat, nlon = u.shape
    best = None

    for rad in radii:
        r0, c0 = i - rad, j - rad
        r1, c1 = i + rad + 1, j + rad + 1
        if r0 < 0 or c0 < 0 or r1 > nlat or c1 > nlon:
            break

        ci, cj = i - r0, j - c0
        uw, vw = u[r0:r1, c0:c1], v[r0:r1, c0:c1]

        psi, ocean = streamfunction(uw, vw, ci, cj, lat_center, dlon, dlat)
        psi = remove_ramp(psi, ocean, ci, cj, lat_center, dlon, dlat)
        eta = equivalent_elevation(psi, true_lat)

        res = _shape(eta, np.hypot(uw, vw), ci, cj, lat_center,
                     dlon, dlat, dz, npts, min_rings)
        if res is None:
            continue
        if best is not None and res["large"]:
            break

        res["rad"] = rad
        for name in ("effective", "speed"):
            res[name + "_row"] += r0
            res[name + "_col"] += c0
        best = res

        if res["large"] or not res["touches"]:
            break

    return best