#!/bin/bash

rm slurm.sh

current_path=${PWD}

cat > slurm.sh <<!
#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH -c $1
#SBATCH --partition=gpu
#SBATCH --time=1:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adam.robinson@nist.gov    
#SBATCH -D $current_path

cd $current_path

module purge
module load powerAI/pytorch-1.5.4
source /opt/DL/pytorch/bin/pytorch-activate
python3 evaluate-parameter-set.py ${@:2}

!

sbatch slurm.sh