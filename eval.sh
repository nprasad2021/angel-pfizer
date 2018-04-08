#!/bin/bash
#SBATCH -n 4
#SBATCH --array=0
#SBATCH --job-name=new_exp
#SBATCH --mem=20GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 03:00:00
#SBATCH --workdir=./subs/
#SBATCH --qos=cbmm

cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/angel-pfizer/eval_final.py