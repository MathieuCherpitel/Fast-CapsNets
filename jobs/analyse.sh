#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/mc2091/thesis/NAS-for-CapsNet/src
## Environment variables
#SBATCH --export=ALL
## Job name
#SBATCH -J analys
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=5-20
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=45634
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#==============================
#  Activate Package Ecosystem
#------------------------------
cd /users/mc2091/thesis/NAS-for-CapsNet/src
conda activate smi

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/jobs/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

#===============================
#  Application launch commands
#-------------------------------

python3 analyse_sol.py

echo "This is an example job. It ran on `hostname -s` (as `whoami`)." >> $RESULTS_DIR/${name}-${i}.output
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" >> $RESULTS_DIR/${name}-${i}.output
echo "Output file has been generated, please check $RESULTS_DIR/${name}-${i}.output"
