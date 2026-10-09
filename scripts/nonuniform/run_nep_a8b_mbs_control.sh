#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.a8b-mbs-control
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export A8B_MBS=2 A8B_TAG=mbs2 A8B_CASE_ORDER="nep healthy" A8B_CASE_TIMEOUT=18m
bash scripts/nonuniform/run_nep_a8b_latent_current.sh
export A8B_MBS=1 A8B_TAG=mbs1 A8B_CASE_ORDER="healthy nep" A8B_CASE_TIMEOUT=13m
bash scripts/nonuniform/run_nep_a8b_latent_current.sh
