#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/mc2091/thesis/NAS-for-CapsNet
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o sbatch/job-%j.output
#SBATCH -e sbatch/job-%j.error
## Job name
#SBATCH -J nas_caps
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=20:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=4000
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight and Conda Environments
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh
conda activate smi

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

#===============================
#  Application launch commands
#-------------------------------
echo "Executing job commands, current working directory is $(pwd)"

NAME="nas_mnist_dmog"
cd src
python3 script.py -n nas_mnist_dmog -d mnist -g 10 -p 10
echo "Job ran on `hostname -s` (as `whoami`)." >> $RESULTS_DIR/$NAME.output
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" >> $RESULTS_DIR/$NAME.output
echo "Output file has been generated, please check $RESULTS_DIR/$NAME.output"
