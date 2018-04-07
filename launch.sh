#!/bin/bash
#SBATCH -n 4
#SBATCH --array=0-18
#SBATCH --job-name=neeraj_exp
#SBATCH --mem=10GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 01:00:00
#SBATCH --workdir=./subs/neeraj/

PATH_ANGEL="/om/user/nprasad/angel-pfizer"

cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/angel-pfizer/main.py $PATH_ANGEL ${SLURM_ARRAY_TASK_ID}