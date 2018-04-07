#!/bin/bash
#SBATCH -n 4
#SBATCH --array=7
#SBATCH --job-name=neeraj_exp
#SBATCH --mem=10GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 00:20:00
#SBATCH --workdir=./subs/neeraj/


cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/angel-pfizer/main.py