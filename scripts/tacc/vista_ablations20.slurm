#!/bin/bash
#SBATCH -J pspe-abl20
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-19
#
# The ablation sweep, rerun with the seed actually reaching the configs.
#
# run_ablations.py passed --seed only to seed_everything, so all five "seeds" of
# the original sweep shared ONE evaluation initial-condition set: six configs
# accept a seed and it was passed to none (defect 46). Fixed 2026-10-02 and
# never rerun, which leaves four ablation claims and section 8.3's dispersion
# resting on intervals five to six times too narrow:
#
#   physics-informed loss on/off
#   operator comparison -- FNO vs DeepONet vs GNOT
#   perception encoder frozen vs fine-tuned
#   faithfulness objective on/off
#
# The contrasts survive (all arms shared the fixed IC set, so the pairing was
# intact); the +/- does not, and every one of those claims quotes one.
#
# The explain arm now also passes max_len=160 rather than taking the 96-token
# default against 136-token reference briefs (defect 47), so the faithfulness
# ablation is measured off the truncation ceiling for the first time. We already
# know from the controls run which way that cuts -- supervised reaches 0.901 and
# the faithfulness term costs 0.396 -- so this is for consistency across
# sections rather than discovery.
#
# One seed per task: a seed that overruns costs that seed, not the experiment.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
OUT=runs/ablations_20
mkdir -p "$OUT"
srun python -u eval/run_ablations.py --testbed dar --grid 64 --seed "$SEED" \
    --device cuda --out "$OUT/seed_${SEED}"
