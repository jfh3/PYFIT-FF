# PYFIT-FF (Description)
Authors: James Hickman (NIST) and Adam Robinson (GMU) 
Description: 
This code uses pytorch to train a Neural network to reproduce a single component system's potential energy surface for use in classical molecular dynamics simulations by approximating energies obtained from density functional theory (DFT) calculations

# Terminology 
The following phrases are helpful for understanding the code 
+ LSP="local structure parameter" (sometimes called Gi's) (each atom gets a LSP, these are the NN inputs)

# Input files
The code require two input files 
1) The DFT database file which contains POSCAR files, DFT energies and optionally DFT forces 
2) An input
If ITRAIN_FORCE==TRUE then 


# Structure objects in PYFIT

		+sid			= structure ID (integer)
		+comment		= comment line in POSCAR
		+scale_factor	= universal scaling factor 
		+a1				= supercell lattice vector-1
		+a2				= supercell lattice vector-2
		+a3				= supercell lattice vector-3
		+V				= supercell volume 
		+N      		= total number of atoms 
		+E				= total energy of structure 
		+v				= volume per atom
		+e				= energy oer atom
		+species		= list of species in structure



# Output files
 
## Dependencies

This code has been tested on linux and macOS however assuming the dependencies below are met the code should run on windows OS as well. 

- PyTorch
- Python 3.x
- numpy
- A Unix OS of some kind.

Usually the following command line will suffice:

```bash
sudo pip3 install torch torchvision numpy
```

# Theory 


A description of the local structural parameters currently implemented can be found at the following link: 
https://www.nature.com/articles/s41467-019-10343-5
