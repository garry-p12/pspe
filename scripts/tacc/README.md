# Running the seed sweep on TACC (Vista / Lonestar6)

The Phase F sweep is embarrassingly parallel — seeds are independent runs — so
on a cluster it costs one seed's wall-clock instead of five. The per-seed output
layout is the same one `eval/run_seeds.py` writes locally, so aggregation
happens back on the laptop with `--aggregate-only` and no cluster-side merge.

## Once per session (on your machine)

    ssh -MNf -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=12h \
        <user>@vista.tacc.utexas.edu

TACC requires password + MFA token, so this login is manual. Every later command
reuses the socket with `-o ControlPath=~/.ssh/cm-%r@%h:%p`.

Better, once you are in: append your laptop's public key to `~/.ssh/authorized_keys`
on the cluster. TACC honours key auth (see the login banner), which removes both
the password and the token from every later connection — no socket, no 12 h
expiry.

## Stage the repo and the dataset

    rsync -az --delete -e "ssh -o ControlPath=~/.ssh/cm-%r@%h:%p" \
        --exclude .git --exclude runs --exclude outputs --exclude '__pycache__' \
        ~/pspe/ <user>@vista.tacc.utexas.edu:'$WORK/pspe/'

`data/pde_testbeds/dar_64.npz` must travel with it: `ensure_dataset` would
otherwise generate it on a miss, and five array tasks missing at once race on
one file path. Shipping the laptop's copy also keeps the GPU numbers comparable
to the CPU ones — same trajectories, only the training stochasticity differs.

## Build the environment (login node — compute nodes have no internet)

    bash scripts/tacc/setup_env.sh

The script keys off `$TACC_SYSTEM` to pick the torch wheel: aarch64 cu126 on
Vista's GH200, x86 cu124 on LS6. It asserts `torch.cuda.is_available()` at the
end — a CPU-only wheel installs and imports happily, and `get_device()` would
fall back to CPU without complaint, so the run would be a CPU run wearing a GPU
label.

## Submit

    sbatch -A <allocation> scripts/tacc/vista_seeds.slurm   # Vista, GH200, -p gh
    sbatch -A <allocation> scripts/tacc/ls6_seeds.slurm     # Lonestar6, A100, -p gpu-a100

Five array tasks, one seed each, 4 h wall-clock cap. Watch with
`squeue -u <user>`; per-task logs land in `pspe-seeds.o<jobid>_<task>`.

## Collect

    rsync -az -e "ssh -o ControlPath=~/.ssh/cm-%r@%h:%p" \
        <user>@vista.tacc.utexas.edu:'$WORK/pspe/runs/seeds_full/' ~/pspe/runs/seeds_full/
    python eval/run_seeds.py --aggregate-only --seeds 0 1 2 3 4 --out runs/seeds_full --full

The aggregate step re-reads each seed's `results.json` and writes
`results_seeds.md` / `.json` with mean ± sample std.
