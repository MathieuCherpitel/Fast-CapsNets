#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/mc2091/thesis/NAS-for-CapsNet
## Environment variables
#SBATCH --export=ALL
## Job name
#SBATCH -J nas-mnist
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

while getopts n:g:p:r: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        g) generations=${OPTARG};;
        p) pop_size=${OPTARG};;
        r) runs=${OPTARG};;
    esac
done

echo "Executing job commands, current working directory is $(pwd)"

for ((i=0; i<runs; i++))
do
    python3 src/script.py -n ${name}_${i} -d mnist -g $generations -p $pop_size > $RESULTS_DIR/${name}_${i}.output
done

echo "This is an example job. It ran on `hostname -s` (as `whoami`)." >> $RESULTS_DIR/mnist-hpc.output
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" >> $RESULTS_DIR/mnist-hpc.output
echo "Output file has been generated, please check $RESULTS_DIR/mnist-hpc.output"
