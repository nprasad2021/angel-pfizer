#!/bin/bash
#SBATCH -n 4
#SBATCH --array=7
#SBATCH --job-name=neeraj_exp
#SBATCH --mem=10GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 00:20:00
#SBATCH --workdir=./subs/neeraj/

module load openmind/singularity/2.4.5
module add openmind/cuda/8.0
module add openmind/cudnn/8.0-5.1

PATH="/om/user/nprasad/angel-pfizer/"

cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/angel-pfizer/main.py $PATH