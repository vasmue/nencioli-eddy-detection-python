"""
Track the eddies of one type with py-eddy-tracker, by overlap of effective
contours.  Run in the py-eddy-tracker environment (environment_tracking.yml),
once for Cyclonic_*.nc and once for Anticyclonic_*.nc -- never both together.
"""

import glob

from py_eddy_tracker.featured_tracking.area_tracker import AreaTracker
from py_eddy_tracker.tracking import Correspondances

files = sorted(glob.glob("/path/to/shapes/Cyclonic_*.nc"))

c = Correspondances(datasets=files, class_method=AreaTracker, virtual=3)
c.track()
c.prepare_merging()
tracks = c.merge(raw_data=False)
tracks.virtual[:] = tracks.time == 0
tracks.filled_by_interpolation(tracks.virtual == 1)

tracks.write_file(filename="tracks_cyclonic.nc")
