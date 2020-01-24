# Background

## Machine learning interatomic potentials 

- The general idea behind most machine learning (ML) interatomic potentials is the following:
	* The local atomic environment (i.e. neighbor list (NBL)) of the i-th atom is quantified as a vector of fixed length (i.e. a structural "descriptor" or "fingerprint" vector). This is done using some set of analytic functions known as local structural parameters (LSP)
		* In PYFIT the LSP vector is denoted Gi (for atom-i)
		* Note: there are many choices for LSP descriptor formula in the literature but the following are popular choices  
			- [Purja-Pun and Mishin](https://www.nature.com/articles/s41467-019-10343-5) 
			- [Behler-Parrinello](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401) 
			- [SOAP](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.184115) 
	* This LSP vector is then used as an input to a Machine learning (ML) regression algorithm (neural network, Gaussian process, etc) which maps from the LSP vector to atomic energy for atom-i (i.e. Gi --> regression --> u_i). The confrontational potential energy of the jth structure of atoms (i.e. the jth POSCAR file) is then predicted to be U_j=sum(u_i)
	* The regression algorithm is trained to interpolate between various pre-computed DFT confrontational energies U_DFT_j by minimization of some objective (for example, RMSE=sqrt(sum((U_j-U_DFT_j)^2/N_j)))
	* Once trained the potential is released to the public to be used in classical atomistic simulations such as molecular dynamics (MD) or Monte-Carlo (MC)
		- Inside the training region the ML potential can have accuracy on the order of DFT itself (<2 meV) but is generally many orders of magnitude faster and has much better scaling to larger systems
		- Outside the training region all purely mathematically ML potential are subject to un-physical extrapolation and therefore should only be used within the region of configuration space for which they trained 


# Terminology 
The following phrases are helpful for understanding the code 
+ LSP="local structure parameter" (sometimes called Gi's) (each atom gets a LSP, these are the NN inputs)


# Primary data structures in PYFIT-FF

## Structure objects 

Because we train to DFT energies for a given configuration (i.e. given POSCAR file). The information for a each structure is stored in a PYFIT structure object which has the following attributes  

+ structure object attributes 
	- sid			= structure ID (integer)
	- comment		= comment line in POSCAR (string)
	- scale_factor	= universal scaling factor  (float) 
	- a1			= supercell lattice vector-1  (3,) np array
	- a2			= supercell lattice vector-2  (3,) np array
	- a3			= supercell lattice vector-3  (3,) np array
	- V				= supercell volume (float)
	- N      		= total number of atoms (int)
	- E				= total energy of structure (float)
	- v				= volume per atom (float)
	- e				= energy oer atom (float)
	- species		= list of species in structure (str or list of string)
	- positions		= Nx3 np array with all atomic positions 


