#!/bin/bash
# Centres and shapes for all regridded days of one depth level.
# Adjust the SBATCH lines, the environment and the paths to your system.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=eddy_detection

dep=$1
if [ -z "$dep" ]; then
    echo "Usage: sbatch $0 <depth_m>" >&2
    exit 1
fi

conda activate eddy_detection
export OMP_NUM_THREADS=1

indir=/path/to/gridded
centers=/path/to/centers/${dep}m
shapes=/path/to/shapes/${dep}m

cd detection
python run_center_detection.py $indir/*/*_${dep}m.nc --out $centers --workers 48
python run_shape_detection.py $indir/*/*_${dep}m.nc --centers $centers \
       --out $shapes --workers 48
