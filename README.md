# PYFIT-FF: 

## Summary

+ Authors: James Hickman (NIST) and Adam Robinson (GMU) 
	* Please send any relevant questions or comments to james.hickman@nist.gov 

+ Description: 
	- PYFIT-FF is a tool for training feed-forward artifical nerual network interatomic potentials. This is done by utalizing the automatic differentiation and optimization library [PYTORCH](https://pytorch.org/) for the optimization process
	-  The main benefits of PYFIT over other NN potentials training tools are the following: 
		* Highly portable 
		* Simple
		* Fast 
		* Flexible
		* Open Source
	- __NOTE__: This README file only contains information on how to install and use PYFIT-FF. For more detailed information see (PYFIT-FF/docs/DOC.md), generally the information in DOC.md would like to get inside the code and modify it.

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
	- see PYFIT-FF/docs/cite.bib for relevant bibtex entries
	
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


## Dependencies installation option-1:    

On a linux machine the dependencies can be installed using 

sudo pip3 install torch torchvision numpy


## Dependencies installation option-2: conda (recommended)  

1) First install "conda" on your machine (conda is a popular open source python package management system)
 	- The conda installation process is well documented at the following link 
 	- https://docs.conda.io/projects/conda/en/latest/user-guide/install/
	- for more information on conda see the following: https://www.youtube.com/watch?v=23aQdrS58e0&t=327s
2) Use the following commands to create a conda environment with the Python 3.x and PyTorch (numpy will be installed automatically as part of this process) 
	- sfdsfdklglds

3) Use the following command to activate this conda environment on your local machine (
	- sfdsfdklglds
	- NOTE: I typically put this command in my ~/.bashrc so that it activates automatically and is essentally my default python implementation


## Input files 


 set.dat",
+ "max_iter"			:	100,
"lambda_rmse"			:	1.0,
"lambda_l1"			:	0.0,
"lambda_l2"			:	0.00001,
"lambda_dU"			:	0.0,
"u_shift"			:	0.795023,
"dump_poscars"			:	false,
"learning_rate"			:	0.05


2) The DFT database file which contains POSCAR files, DFT energies



# Output files
 
