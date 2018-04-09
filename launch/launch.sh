#!/bin/bash
#SBATCH -n 4
#SBATCH --array=0-2
#SBATCH --job-name=new_exp
#SBATCH --mem=10GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 03:00:00
#SBATCH --workdir=./subs/ensemble/
#SBATCH --qos=cbmm

PATH_ANGEL="/om/user/nprasad/angel-pfizer"

cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/angel-pfizer/main.py $PATH_ANGEL ${SLURM_ARRAY_TASK_ID}