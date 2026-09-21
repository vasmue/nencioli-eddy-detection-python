"""
Nencioli et al. (2010) vector-geometry eddy detection.

Reads gridded u/v from netCDF and needs nothing else -- it does not assume the
FESOM regridding, or any particular grid.

ARRAY ORIENTATION
    Everything here works in (lat, lon): a row is an East-West section at
    constant latitude, which is the section the v-reversal constraint is
    applied along.

MASK CONVENTION
    build_region_mask returns true INSIDE the region of interest.  uv_search
    takes the opposite: true where the point is to be excluded.

ROTATION SENSE vs EDDY TYPE
    uv_search reports the sense of rotation, +1 counter-clockwise and -1
    clockwise, which is all the velocity field can tell you.  Whether that is
    cyclonic or anticyclonic depends on the hemisphere, so the conversion
    needs geographic latitude and happens outside this module.
"""

import numpy as np


def _euler_matrix(alpha, beta, gamma):
    a, b, g = np.radians([alpha, beta, gamma])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    return np.array([
        [cg * ca - sg * cb * sa,  cg * sa + sg * cb * ca,  sg * sb],
        [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
        [sb * sa,                 -sb * ca,                cb],
    ])


def rotated_to_geographic(alpha, beta, gamma, rlon, rlat):
    """Map rotated-frame coordinates to geographic ones."""
    R = np.linalg.inv(_euler_matrix(alpha, beta, gamma))

    rlon = np.radians(rlon)
    rlat = np.radians(rlat)
    xr = np.cos(rlat) * np.cos(rlon)
    yr = np.cos(rlat) * np.sin(rlon)
    zr = np.sin(rlat)

    x = R[0, 0] * xr + R[0, 1] * yr + R[0, 2] * zr
    y = R[1, 0] * xr + R[1, 1] * yr + R[1, 2] * zr
    z = R[2, 0] * xr + R[2, 1] * yr + R[2, 2] * zr

    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1, 1)))


def build_region_mask(true_lon, true_lat, regions):
    """True inside the region of interest.  Boxes are OR-ed."""
    if not regions:
        return np.ones(true_lat.shape, dtype=bool)

    mask = np.zeros(true_lat.shape, dtype=bool)
    for r in regions:
        lo0, lo1 = r["lon"]
        la0, la1 = r["lat"]
        mask |= ((true_lon >= lo0) & (true_lon <= lo1) &
                 (true_lat >= la0) & (true_lat <= la1))
    return mask


# ---------------------------------------------------------------------------
# The four velocity constraints (uv_search.m)
# ---------------------------------------------------------------------------

def _quadrants(ub, vb):
    """Quadrant of each boundary vector: 1 = NE, 2 = NW, 3 = SW, 4 = SE."""
    q = np.zeros(ub.shape, dtype=np.int32)
    q[(ub >= 0) & (vb >= 0)] = 1
    q[(ub < 0) & (vb >= 0)] = 2
    q[(ub < 0) & (vb < 0)] = 3
    q[(ub >= 0) & (vb < 0)] = 4
    return q


def _rotates_coherently(u_small, v_small):
    """Fourth constraint: the velocity vector turns monotonically through a
    full circuit around the centre.

    The circuit is traversed counter-clockwise: along increasing longitude at
    the lowest latitude, up the eastern edge, back along the highest latitude,
    down the western edge, closing on the starting cell.

    The test is independent of rotation sense: going once counter-clockwise
    around the boundary, the velocity winds through 2*pi for both cyclones
    and anticyclones, since the velocity is the radius vector turned by
    +/-90 degrees and a fixed rotation does not change the winding direction.
    """
    ub = np.concatenate([u_small[0, :], u_small[1:, -1],
                         u_small[-1, -2::-1], u_small[-2::-1, 0]])
    vb = np.concatenate([v_small[0, :], v_small[1:, -1],
                         v_small[-1, -2::-1], v_small[-2::-1, 0]])

    q = _quadrants(ub, vb)

    spin = np.flatnonzero(q == 4)
    if spin.size == 0 or spin.size == q.size:
        return False

    if spin[0] == 0:
        # circuit starts in the fourth quadrant: unwrap from the first vector
        # that is not, so the +4 offset is applied to the right stretch
        start = np.flatnonzero(q != 4)[0]
    else:
        start = spin[-1] + 1

    q = q.copy()
    q[start:] += 4

    dq = np.diff(q)
    return not (np.any(dq > 1) or np.any(dq < 0))


def uv_search(u, v, lon, lat, mask, a, b):
    """Find the points satisfying all four velocity constraints.

    u, v, lon, lat, mask are all 2-D in (lat, lon).  mask is True where the
    point is OUTSIDE the region of interest.

    Returns a dict with arrays lat, lon, sense, i (lat index), j (lon index).
    sense is +1 for counter-clockwise rotation and -1 for clockwise.

    Constraints 1 and 2 are comparisons between shifted copies of u and v, so
    they are evaluated over the whole grid at once.  Constraints 3 and 4 are
    local searches and run only on the points that survive.
    """
    nlat, nlon = v.shape
    borders = max(a, b) + 1

    u = np.array(u, dtype=np.float64, copy=True)
    v = np.array(v, dtype=np.float64, copy=True)

    vel2 = u ** 2 + v ** 2

    bad = mask | (vel2 == 0) | (np.abs(u) > 1e10) | (np.abs(v) > 1e10)
    u[bad] = np.nan
    v[bad] = np.nan
    vel2[bad] = np.nan

    # --- constraint 1: reversal of v along an East-West section -------------
    s = np.sign(v)
    ds = s[:, 1:] - s[:, :-1]
    cross = (ds != 0) & ~np.isnan(ds)          # crossing between j and j+1

    # keep away from the domain edges so every stencil below is in range
    valid = np.zeros_like(cross)
    valid[borders - 1:nlat - borders + 1, borders - 1:nlon - borders] = True
    cross &= valid

    ii, jj = np.nonzero(cross)
    if ii.size == 0:
        return _empty_result()

    va = v[ii, jj]
    vb = v[ii, jj + 1]
    v_left = v[ii, jj - a]
    v_right = v[ii, jj + 1 + a]

    anti = (va >= 0) & (v_left > va) & (v_right < vb)
    cyc = (va < 0) & (v_left < va) & (v_right > vb)

    keep = anti | cyc
    ii, jj = ii[keep], jj[keep]
    sense = np.where(anti[keep], -1, 1).astype(np.int32)

    if ii.size == 0:
        return _empty_result()

    # --- constraint 2: reversal of u along a North-South section ------------
    def _u_ok(col):
        um_a = u[ii - a, col]
        um_1 = u[ii - 1, col]
        up_a = u[ii + a, col]
        up_1 = u[ii + 1, col]
        ok_anti = ((um_a <= 0) & (um_a <= um_1) &
                   (up_a >= 0) & (up_a >= up_1))
        ok_cyc = ((um_a >= 0) & (um_a >= um_1) &
                  (up_a <= 0) & (up_a <= up_1))
        return np.where(sense == -1, ok_anti, ok_cyc)

    keep = _u_ok(jj) | _u_ok(jj + 1)
    ii, jj, sense = ii[keep], jj[keep], sense[keep]

    if ii.size == 0:
        return _empty_result()

    # --- constraints 3 and 4, per surviving candidate -----------------------
    out_i, out_j, out_sense = [], [], []
    d = a - 1

    for i0, j0, rot in zip(ii, jj, sense):
        # third constraint: the centre is a local minimum of speed.
        # Since sqrt() is monotonic, vel2 = u² + v² has the same minima
        # as the actual speed.
        box = vel2[i0 - b:i0 + b + 1, j0 - b:j0 + b + 2]
        if np.all(np.isnan(box)):
            continue

        # Find the minimum-speed cell in the box.
        # where several cells share the minimum speed, any of them will do
        di, dj = np.unravel_index(np.nanargmin(box), box.shape)
        ic = i0 - b + di
        jc = j0 - b + dj

        # That minimum can sit on the edge of the box, with something smaller
        # just outside it -- smallest in the box, but not a local minimum.
        # Search again in a box of the same size centred on it: if the
        # minimum is unchanged, nothing smaller is adjacent and the point is
        # a genuine local minimum.
        around = vel2[max(ic - b, 0):min(ic + b, nlat - 1) + 1,
                      max(jc - b, 0):min(jc + b, nlon - 1) + 1]
        if np.nanmin(around) != vel2[ic, jc]:
            continue

        # fourth constraint: the velocity must turn coherently around a
        # circuit of radius a-1 centred on the minimum
        rows = slice(max(ic - d, 0), min(ic + d, nlat - 1) + 1)
        cols = slice(max(jc - d, 0), min(jc + d, nlon - 1) + 1)
        u_small = u[rows, cols]
        v_small = v[rows, cols]

        # a circuit that reaches land or the sea floor cannot be evaluated
        if np.isnan(u_small).any() or np.isnan(v_small).any():
            continue

        if _rotates_coherently(u_small, v_small):
            out_i.append(ic)
            out_j.append(jc)
            out_sense.append(rot)

    return _finalise(out_i, out_j, out_sense, lon, lat)


def _empty_result():
    return {k: np.array([], dtype=t) for k, t in
            (("lat", float), ("lon", float), ("sense", np.int32),
             ("i", np.int32), ("j", np.int32))}


def _finalise(out_i, out_j, out_sense, lon, lat):
    """Deduplicate centres.

    Several velocity reversals can resolve to the same minimum, so the same
    centre may be recorded more than once.  Sort by grid index and keep one
    entry per cell.
    """
    if not out_i:
        return _empty_result()

    i = np.asarray(out_i, dtype=np.int64)
    j = np.asarray(out_j, dtype=np.int64)
    t = np.asarray(out_sense, dtype=np.int32)

    order = np.lexsort((j, i))
    i, j, t = i[order], j[order], t[order]

    first = np.ones(i.size, dtype=bool)
    first[1:] = (np.diff(i) != 0) | (np.diff(j) != 0)
    i, j, t = i[first], j[first], t[first]

    return {
        "lat": lat[i, j],
        "lon": lon[i, j],
        "sense": t,
        "i": i.astype(np.int32),
        "j": j.astype(np.int32),
    }
