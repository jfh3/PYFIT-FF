# PYFIT-FF 
- Authors: James Hickman (NIST) and Adam Robinson (GMU) 
+ Description: 
	- This code uses pytorch to train a Neural network interatomic potential to reproduce a single component system's potential energy surface for use in classical molecular dynamics simulations by approximating energies obtained from density functional theory (DFT) calculations
	

# Installation

Nessesary dependencies (see below for dependency installation instructions):  

- PyTorch
- Python 3.x
- Numpy

Once the dependencies are met then PYFIT can be installed using the following instructions 

1) Use the following command to get the PYFIT-FF source code from Github
 	- git clone https://github.com/jfh3/PYFIT-FF
	- or manually use the green "download or clone" button on https://github.com/jfh3/PYFIT-FF
 	- This will make a directory called "PYFIT-FF" on your machine 

2) Use the following commands to make PYFIT executable from any directory  
 	- mv PYFIT-FF ~/bin
	- cd ~/bin/
	- ln -s PYFIT-FF/src/pyfit.py  PYFIT 
	- Note: because pyfit.py has the line #!/usr/bin/env python3 it can be run like executable and therefore your system will find PYFIT in ~/bin and add it to the PATH  

3) Run PYFIT using the provided example to get started 
	- cd PYFIT-FF/example/
	- pyfit input.json 

+ Additional comments 
	- This code has been tested on linux and macOS systems, however, assuming the following dependencies are met for your python implementation then the code should run on windows OS as well.
	- Note techincally PYFIT can be run directly from

## Dpendenceies installation option-1: conda (recommended)  

1) First install "conda" on your machine (conda is a popular open source python package management system)
 	- The conda installation process is well documented at the following link 
 	- https://docs.conda.io/projects/conda/en/latest/user-guide/install/
	- for more information on conda see the following: https://www.youtube.com/watch?v=23aQdrS58e0&t=327s
2) Use the following commands to create a conda enviroment with the Python 3.x and PyTorch (numpy will be installed automatically as part of this process) 
	- sfdsfdklglds

3) Use the following command to activate this conda enviroment on your local machine (
	- sfdsfdklglds
	- NOTE: I typically put this command in my ~/.bashrc so that it activates automatically and is essentally my default python implementation

 



## Install dependencies 



Usually the following command line will suffice:

sudo pip3 install torch torchvision numpy

On a linux machine the following commands will allow to to run pyfit from anywhere on your computer using 




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
 


# Future planned updates

- multicomponent neural network interatomic potentials using Mishin descriptors
- implement Behler-Parrinello descriptors
- single component PINN interatomic potential training
- multicomponent component PINN interatomic potential training
- implement force fitting for select potential options (maybe)

# Theory 

A description of the local structural parameters currently implemented can be found at the following link: 
https://www.nature.com/articles/s41467-019-10343-5
