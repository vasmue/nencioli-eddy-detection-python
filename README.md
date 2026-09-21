# Nencioli eddy detection in Python

A Python version of the vector-geometry eddy detection of Nencioli et al.
(2010), with a preparation stage that brings unstructured FESOM2 output onto a
regular grid, and output that [py-eddy-tracker](https://github.com/AntSimi/py-eddy-tracker)
can track directly.

The detection works on any gridded u/v field on a lon/lat grid: regridded model
output, CMEMS/AVISO geostrophic velocities, or anything else.

## Layout

```
regridding/     FESOM2 output -> regular grid, rotated or not      (config.yaml)
detection/      eddy centres, eddy shapes, files for tracking      (detection.yaml)
examples/       SLURM scripts and a tracking script
```

The two directories are independent.  Each has its own config file, which the
scripts read from their own directory.

## Workflow

### 1. Regridding (FESOM2 output only)

Set `mode` in `regridding/config.yaml`:

| mode            | input                  | target grid                                                  |
|-----------------|------------------------|--------------------------------------------------------------|
| `rotate_regrid` | FESOM2 `unod`/`vnod`   | regular grid in a frame with the pole moved onto the equator  |
| `regrid`        | FESOM2 `unod`/`vnod`   | regular geographic grid                                       |
| `none`          | already on a lon/lat grid | nothing to do; go straight to the detection                |

`rotate_regrid` is for polar domains: rotating the pole onto the equator keeps
the grid spacing nearly uniform and removes the pole singularity.  The domain
must then be a whole polar cap.  `regrid` is for everything else.

```
cd regridding
python precompute_weights.py                          # once per mesh and grid
python validate_regrid.py YEAR DEPTH                  # once, checks the weights
python rotate_regrid_uv_parallel.py YEAR DEPTH --workers 8
```

`precompute_weights.py` builds the interpolation weights once; every later run
only reads them.  The mode, rotation and grid are stored with the weights, and
any run with a config that no longer matches them stops with an error.

### 2. Detection

Set the input variable names, the grid (rotated or not) and the region in
`detection/detection.yaml`, then:

```
cd detection
python run_center_detection.py /path/to/gridded/*/*.nc --workers 48
python run_shape_detection.py  /path/to/gridded/*/*.nc --centers /path/to/centers --workers 48
```

Both take the list of daily input files and write one output file per day.
They resume where they left off unless `--overwrite` is given.

**Centres** are found with the four velocity constraints of Nencioli et al.
(2010).  Each centre gets its rotation sense and, from the geographic latitude,
its type (cyclonic or anticyclonic), so both hemispheres work.

**Shapes** are computed per centre from a streamfunction, converted to an
equivalent surface elevation and contoured at a fixed step `dz`.  Each day
produces:

- `eddy_shapes_YYYY_DDD.nc`: every shaped eddy, with its effective and speed
  contours, radii, amplitude, speed, shape error and the flags below.
- `Cyclonic_YYYY_DDD.nc` and `Anticyclonic_YYYY_DDD.nc`: the eddies to track,
  in py-eddy-tracker's format.  Eddies that are `nested`, `too_small` or have
  a `weak_shape`, or whose shape error is `tracking.max_err` or more, are left
  out.

Flags in `eddy_shapes_*.nc`:

- `weak_shape`: no closed contour had speed increasing outward at its four
  extremes, so the shape is only the largest closed contour.
- `nested`: the centre lies inside the effective contour of a larger eddy of
  the same type, i.e. it is a secondary speed minimum within that eddy.
- `too_small`: the effective radius is below `tracking.min_rad`.

### Choosing `dz`

The contour step depends strongly on the data: eddies in altimetry are an
order of magnitude stronger than eddies below the surface in an ice-covered
ocean.  For one day of data,

```
python check_dz.py UV_FILE CENTERS_FILE
```

starts from the typical elevation range around the centres, halves the step
until the results stop changing, and recommends the coarsest step that already
agrees with the next finer one.  During a run, the shape stage also reports
the median number of contours per eddy and warns when `dz` looks too coarse or
too fine.

### Checking a day

```
python plot_day.py UV_FILE SHAPES_FILE [--lon MIN MAX] [--lat MIN MAX]
```

draws speed, velocity arrows and the effective contours: blue for cyclones,
red for anticyclones, dashed for weak shapes.
![Eddies detected on one day: speed, velocity, cyclones in blue, anticyclones in red](docs/example_day.png)

### 3. Tracking

Tracking is done by py-eddy-tracker, in its own environment.  See
`examples/track.py`.  Track cyclones and anticyclones separately.

For a rotated grid the positions and contours are in the rotated frame.  A
rotation of the sphere preserves distances and areas, so tracking links the
same eddies as it would in geographic coordinates.

## Differences from the Matlab version

The centre detection follows the Matlab code.  The shape stage keeps its logic
-- the largest closed contour across which speed increases, a window that grows
while the contour reaches its edge -- with these changes:

- The streamfunction is the least-squares solution of a Poisson problem with
  natural boundary conditions, instead of an average of two path integrals.
- Contours are drawn at a fixed step in metres of equivalent elevation,
  instead of 100 levels spread over each window, so levels mean the same for
  every eddy.
- Windows are solved on the ocean cells only, instead of being shrunk until
  they contain no land.
- Eddies are flagged instead of dropped, so the selection for tracking can be
  changed without rerunning the detection.

## Environments

```
conda env create -f environment.yml            # regridding and detection
conda env create -f environment_tracking.yml   # py-eddy-tracker
```

py-eddy-tracker pins old versions of numpy and numba, so it lives in its own
environment.  `pyfesom2` is needed only for the regridding.

## Reference and licence

Nencioli, F., C. Dong, T. Dickey, L. Washburn and J. C. McWilliams (2010): A
vector geometry–based eddy detection algorithm and its application to a
high-resolution numerical model product and high-frequency radar surface
velocities in the Southern California Bight.  *J. Atmos. Oceanic Technol.*,
27, 564–579.

The detection is a port of the Matlab code by Francesco Nencioli and Charles
Dong, which is released under the GNU General Public License v3.  This code is
therefore also released under the GPL v3; see `LICENSE`.
