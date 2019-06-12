#!/bin/sh

rm slurm.sh

current_path=${PWD}

cat > slurm.sh <<!
#!/bin/sh
#SBATCH -c 52
#SBATCH --partition=debug
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adam.robinson@nist.gov    
#SBATCH -D $current_path

cd $current_path

module purge
module load powerAI/pytorch-1.5.4
source /opt/DL/pytorch/bin/pytorch-activate

for start in \$(seq 0 51)
do 
let "end = \$start + 1"
python3 run-correlation-analysis.py cmd_log_\$start test_output/ \$start:\$end input/EOS/EOS-E-full-lsparam.dat 0.02 0.001 --cpu -n 1 -u &
done

wait

!

sbatch slurm.sh