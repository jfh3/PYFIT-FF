#!/bin/bash
#SBATCH --partition=mml  # -p, partition
#SBATCH --time 60:00:00    # -t, time (hh:mm:ss or dd-hh:mm:ss)
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH -o log2
#SBATCH -J 01 #job name
#SBATCH -D /wrk/jfh3/05-30-19-TEST-GANGAS-PERT-PINN-CODE/Rc4.5-Tc1.0-Gi2-EOS-NN  #set working directory of batch script before execution

cd /wrk/jfh3/05-30-19-TEST-GANGAS-PERT-PINN-CODE/Rc4.5-Tc1.0-Gi2-EOS-NN  

module load intel/2015 openmpi/1.10.2/intel-15
#module unload intel/2015 openmpi/2.1.0/intel-15
#module load intel/2017 openmpi/2.1.0/intel-17
mpirun /home/jfh3/bin/05-30-19-fit-intel15-LS-MOD < input.dat > log

#mpirun /home/jfh3/bin/fit-cleaned-02-07-19 < input.dat > log

