#!/bin/bash
#SBATCH -J pspe-prep
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#
# Generate the swe/rdf datasets at 64^2 once, before the transfer array starts.
# The array's five tasks each need all three families; if the files are absent
# they would all generate them at once and race on the same .npz paths.
#
#   sbatch -A <allocation> scripts/tacc/vista_prep_data.slurm
set -euo pipefail

VENV="${VENV:-$WORK/pspe-venv}"
REPO="${REPO:-$WORK/pspe}"

module purge
module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"
cd "$REPO"

for FAMILY in swe rdf; do
    if [ -f "data/pde_testbeds/${FAMILY}_64.npz" ]; then
        echo "$FAMILY already present, skipping"
        continue
    fi
    srun python scripts/generate_data.py testbed="$FAMILY" data.grid=64 device=cuda
done
ls -lh data/pde_testbeds/
