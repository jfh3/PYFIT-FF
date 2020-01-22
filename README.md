# Background: Machine learning interatomic potentials 

- The general idea behind most machine learning (ML) interatomic potentials is the following:
	* The local atomic environment (i.e. neighbor list (NBL)) of the i-th atom is quantified as a vector of fixed length (i.e. a structural "descriptor" or "fingerprint" vector). This is done using some set of analytic functions known as local structural parameters (LSP)
		* In PYFIT the LSP vector is denoted Gi (for atom-i)
		* Note: there are many choices for LSP descriptor formula in the literature but the following are popular choices  
			- [Purja-Pun and Mishin](https://www.nature.com/articles/s41467-019-10343-5) 
			- [Behler-Parrinello](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401) 
			- [SOAP](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.184115) 
	* This LSP vector is then used as an input to a Machine learning (ML) regression algorithm (neural network, Gaussian process, etc) which maps from the LSP vector to atomic energy for atom-i (i.e. Gi --> regression --> u_i). The configurational potential energy of the jth structure of atoms (i.e. the jth POSCAR file) is then predicted to be U_j=sum(u_i)
	* The regression algorithm is trained to interpolate between various pre-computed DFT configurational energies U_DFT_j by minimization of some objective (for exmaple, RMSE=sqrt(sum((U_j-U_DFT_j)^2/N_j)))
	* Once trained the potential is released to the public to be used in classical atomistic simulations such as molecular dynamics (MD) or Monte-Carlo (MC)
		- Inside the training region the ML potential can have accuracy on the order of DFT itself (<2 meV) but is generally many orders of magnetude faster and has much better scaling to larger systems
		- Outside the training region all purely mathematically ML potential are subject to un-physical extrpolation and therefore should only be used within the region of configuration space for which they trained 

# PYFIT-FF 

## Summary 

- Authors: James Hickman (NIST) and Adam Robinson (GMU) 

+ Description: 
	- PYFIT-FF is a tool for training ML potentials but for the special case in which the regression function is a feed-forward artifical nerual network. To acheive this the code uses the automatic differentiation and optimization library [PYTORCH](https://pytorch.org/) for the optimization process
	-  The main benefits of PYFIT over other NN potentials training tools are the following: 
		* Highly portable 
		* Fast 
		* Flexible
		* Open Source
	- This README file currently contains all documentation, if something is unclear please email the author at james.hickman@nist.gov 

## Current functionality

+ Single component mathematical NN training using the local atomic environment descriptors developed by [Purja-Pun and Mishin](https://www.nature.com/articles/s41467-019-10343-5) 

## Planned updates

We are actively working on extending the PYFIT functionality to the following cases (in more or less chronological order) 

+ single component PINN interatomic potential training
+ multicomponent neural network interatomic potentials using the [Purja-Pun and Mishin](https://www.nature.com/articles/s41467-019-10343-5) descriptors 
+ multicomponent neural network interatomic potentials using the Behler-Parrinello descriptors
+ multicomponent component PINN interatomic potential training

## Citing PYFIT-FF 

If you use PYFIT-FF to generate an interatomic potential used in a publication please use the relevant citation
	

# Installation

Necessary dependencies (see below for dependency installation instructions):  

- PyTorch
- Python 3.x
- Numpy

Once the dependencies are met then PYFIT can be installed using the following instructions 

1) Use the following command to get the PYFIT-FF source code from Github
 	- git clone https://github.com/jfh3/PYFIT-FF
	- Alternatively you can  manually use the green "clone or download" button on https://github.com/jfh3/PYFIT-FF
 	- This will make a directory called "PYFIT-FF" on your machine 

2) Use the following commands to make PYFIT executable from any directory on your machine  
 	- mv PYFIT-FF ~/bin
	- cd ~/bin/
	- ln -s PYFIT-FF/src/pyfit.py  PYFIT 

Once these commands are run your system should automatically find the link to PYFIT in ~/bin/ and add it to the PATH which can then be called from the command line. Because pyfit.py has the line #!/usr/bin/env python3 it can be run like executable 

3) Run PYFIT using the provided example to get started 
	- cd PYFIT-FF/example/
	- PYFIT input.json 
	- Note: The command "PYFIT input.json" points the link in ~/bin/ and executes the program. An alternative is to run pyfit directly from the /src/ directory 
		* cd PYFIT-FF/src/
		* python3.7 pyfit.py input.json 

+ Additional comments 
	- This code has been tested on Linux and macOS systems, however, assuming the following dependencies are met for your python implementation then the code should run on windows OS as well.
	- Note technically PYFIT can be run directly from

## Dependencies installation option-1: conda (recommended)  

1) First install "conda" on your machine (conda is a popular open source python package management system)
 	- The conda installation process is well documented at the following link 
 	- https://docs.conda.io/projects/conda/en/latest/user-guide/install/
	- for more information on conda see the following: https://www.youtube.com/watch?v=23aQdrS58e0&t=327s
2) Use the following commands to create a conda environment with the Python 3.x and PyTorch (numpy will be installed automatically as part of this process) 
	- sfdsfdklglds

3) Use the following command to activate this conda environment on your local machine (
	- sfdsfdklglds
	- NOTE: I typically put this command in my ~/.bashrc so that it activates automatically and is essentally my default python implementation


## Install dependencies 



Usually the following command line will suffice:

sudo pip3 install torch torchvision numpy

On a Linux machine the following commands will allow to to run PYFIT from anywhere on your computer using 




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

Because we train to DFT energies for a given configuration (i.e. given POSCAR file). The information for a each structure is stored in a PYFIT structure object which has the following attributes  

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
 
