# PYFIT-FF (Description)
Authors: James Hickman (NIST) and Adam Robinson (GMU) 
Description: 
This code uses pytorch to train a Neural network to reproduce a single component system's potential energy surface for use in classical molecular dynamics simulations by approximating energies obtained from density functional theory (DFT) calculations

# Installation

This code has been tested on linux and macOS however assuming the dependencies below are met the code should run on windows OS as well. 

## Installation option-1: conda (recommended)  

1) Install the python open source package management system "conda" on your machine 
	-The installation process is well documented at the following link 
	-https://docs.conda.io/projects/conda/en/latest/user-guide/install/



## Install dependencies 


- PyTorch
- Python 3.x
- numpy
- A Unix OS of some kind.

Usually the following command line will suffice:

sudo pip3 install torch torchvision numpy

On a linux machine the following commands will allow to to run pyfit from anywhere on your computer using 

ln -s PYFIT-FF/src/pyfit.py ~/bin/PYFIT 

Note: This will then act almost like an executable because pyfit.py has the line #!/usr/bin/env python3


# Terminology 
The following phrases are helpful for understanding the code 
+ LSP="local structure parameter" (sometimes called Gi's) (each atom gets a LSP, these are the NN inputs)

# Input files
PYFIT-FF requires two input files 

1) A json file with the following input parameters (typically called input.json)

NOTE: See examples directory for 

+ "nn_file_path"			:	"nn0.dat",
+ "dataset_path"			:	"data-set.dat",
+ "max_iter"			:	100,
"lambda_rmse"			:	1.0,
"lambda_l1"			:	0.0,
"lambda_l2"			:	0.00001,
"lambda_dU"			:	0.0,
"u_shift"			:	0.795023,
"dump_poscars"			:	false,
"learning_rate"			:	0.05


2) The DFT database file which contains POSCAR files, DFT energies


# Primary data structures in PYFIT-FF

## Structure objects 

Because we train to DFT energies for a given configuration (i.e. given POSCAR file). The information for a each structur is stored in a PYFIT structure object which has the following attributes  

+ structure object attributes 
	- sid			= structure ID (integer)
	- comment		= comment line in POSCAR (string)
	- scale_factor	= universal scaling factor  (float) 
	- a1			= supercell lattice vector-1 (3x1 np array)
	- a2			= supercell lattice vector-2
	- a3			= supercell lattice vector-3
	- V				= supercell volume 
	- N      		= total number of atoms 
	- E				= total energy of structure 
	- v				= volume per atom
	- e				= energy oer atom
	- species		= list of species in structure



# Output files
 




# TODO

+ implement force fitting (maybe)
+ add PINN training functionally to current PYFIT code


# Theory 


A description of the local structural parameters currently implemented can be found at the following link: 
https://www.nature.com/articles/s41467-019-10343-5
