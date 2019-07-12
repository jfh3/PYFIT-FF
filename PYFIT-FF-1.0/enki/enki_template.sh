#!/bin/bash

rm {{{job_name}}}.sh

current_path=${PWD}

cat > {{{job_name}}}.sh <<!
#!/bin/sh
#SBATCH --gres=gpu:{{{n_gpu}}}
#SBATCH -c {{{n_cores}}}
#SBATCH --partition={{{partition}}}
#SBATCH --time={{{time}}}  
#SBATCH -D $current_path

cd $current_path

module purge
module load powerAI/pytorch-1.5.4
source /opt/DL/pytorch/bin/pytorch-activate
{{{command}}}

!

sbatch {{{job_name}}}.sh