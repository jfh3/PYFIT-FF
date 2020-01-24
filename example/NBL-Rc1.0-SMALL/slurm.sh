#!/bin/bash
#SBATCH --partition=mml  # -p, partition
#SBATCH --time 60:00:00    # -t, time (hh:mm:ss or dd-hh:mm:ss)
#SBATCH --ntasks-per-node=16 
#SBATCH --mem=24000
#SBATCH --nodes=1
#SBATCH -o log2
#SBATCH -J test #job name
#SBATCH -D /wrk/jfh3/2020-01-22-PYFIT-TEST/NBL-Rc1.0-SMALL  #set working directory of batch script before execution

cd /wrk/jfh3/2020-01-22-PYFIT-TEST/NBL-Rc1.0-SMALL  

#module purge
#module load intel/2015 openmpi/1.10.2/intel-15

#module unload intel/2015 openmpi/2.1.0/intel-15
#module load intel/2017 openmpi/2.1.0/intel-17
mpirun fit-3.4.5-NBL  < input.dat > log
#mpirun fit-3.4.5 < input.dat > log

