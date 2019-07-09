#!/bin/bash

rm slurm.sh

current_path=${PWD}

cat > slurm.sh <<!
#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH -c $1
#SBATCH --partition=gpu
#SBATCH --time=8:30:00  
#SBATCH -D $current_path

cd $current_path

module purge
module load powerAI/pytorch-1.5.4
source /opt/DL/pytorch/bin/pytorch-activate
time python3 coevolution_eval.py ${@:2:99}

!

sbatch slurm.sh