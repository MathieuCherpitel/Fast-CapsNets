#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/mc2091/thesis/NAS-for-CapsNet
## Environment variables
#SBATCH --export=ALL
## Job name
#SBATCH -J nas-mnist
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=3-20
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
cd /users/mc2091/thesis/NAS-for-CapsNet
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

echo "Executing job commands, current working directory is $(pwd)"

python3 src/script.py -n mnist0-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist0-hpc.output
python3 src/script.py -n mnist1-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist1-hpc.output
python3 src/script.py -n mnist2-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist2-hpc.output
python3 src/script.py -n mnist4-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist4-hpc.output
python3 src/script.py -n mnist5-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist5-hpc.output
python3 src/script.py -n mnist6-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist6-hpc.output
python3 src/script.py -n mnist7-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist7-hpc.output
python3 src/script.py -n mnist8-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist8-hpc.output
python3 src/script.py -n mnist9-hpc -d mnist -g 10 -p 10 > $RESULTS_DIR/mnist9-hpc.output

echo "This is an example job. It ran on `hostname -s` (as `whoami`)." >> $RESULTS_DIR/mnist-hpc.output
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" >> $RESULTS_DIR/mnist-hpc.output
echo "Output file has been generated, please check $RESULTS_DIR/mnist-hpc.output"
