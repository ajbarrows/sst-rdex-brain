#!/bin/sh

#SBATCH --job-name=mapall
#SBATCH --output=job_logs/%x_%j.out
#SBATCH --error=job_logs/%x_%j.err
#SBATCH --mail-type=ALL

#SBATCH --partition=bigmemwk
#SBATCH --time=6-6:00:00

#SBATCH --mem=768G

#SBATCH --cpus-per-task=16


export OMP_NUM_THREADS=1

source ${HOME}/.bashrc

conda activate sst-rdex

cd ${SLURM_SUBMIT_DIR}
cd ../

set -x

echo "Submission Dir:  ${SLURM_SUBMIT_DIR}"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"

kedro run --pipeline=map_allregressor