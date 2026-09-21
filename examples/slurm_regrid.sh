#!/bin/bash
# Regrid one depth level for several years, one year per task.
# Adjust the SBATCH lines, the environment and the years to your system.
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=regrid

dep=$1
if [ -z "$dep" ]; then
    echo "Usage: sbatch $0 <depth_m>" >&2
    exit 1
fi

conda activate eddy_detection
export OMP_NUM_THREADS=1

cd regridding
for yy in {2015..2020}; do
    srun --exact --ntasks=1 --cpus-per-task=8 \
         python rotate_regrid_uv_parallel.py $yy $dep --workers 8 &
done
wait
