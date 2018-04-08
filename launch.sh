#!/bin/bash
#SBATCH -n 4
#SBATCH --array=6
#SBATCH --job-name=all_exp
#SBATCH --mem=10GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 02:00:00
#SBATCH --workdir=./subs/all/

PATH_ANGEL="/om/user/nprasad/angel-pfizer"

cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/angel-pfizer/main.py $PATH_ANGEL ${SLURM_ARRAY_TASK_ID}