#include <stdlib.h>
#include <math.h>

#include "globals.h"

// global log file
FILE* logout;

int errcode;

// MPI related
int me, nprocs;
MPI_Comm world;

//
// --------------------- Analytical derivative related variables -----------
//
double *local_partial_deriv;   // Array to store partial derivatives of local BO parameters
			       // w.r.t. fitting parameters for all atoms in each cpus.
double *global_partial_deriv;  // Array to store partial derivatives of local BO
			       // parameters w.r.t. fitting parameters for all atoms.
// Array to store locally sums of partial derivatives w.r.t.
// fitting parameters for structures.
//  _______ _______ _______ _______
// |dE/dp_1|dE/dp_2| ..... |dE/dp_n| <- structure 0
// |dE/dp_1|dE/dp_2| ..... |dE/dp_n| <- structure 1
//  ...............................   .......
// |dE/dp_1|dE/dp_2| ..... |dE/dp_n| <- structure N-1
//  ------- ------- ----- -------
//
// Size of the array is (number of structures x number of fitting parameters).
double *local_sum_partial_deriv;  // Array to store locally sums of partial derivatives of structures.
double *global_sum_partial_deriv; // Array to store globally sums of partial derivatives for structures.
				  // Size of the array is same as the size of the previous array.
// Partial derivatives related to regularization of local BO parameters.
double *local_sum_hbconst_partial_deriv;   // For local sum.
double *global_sum_hbconst_partial_deriv;  // For global sum.
int isGlobalGrad; // Flag to choose method of derivative calculations only relevant to root:
		  // 0 for finite difference.
		  // 1 for analytical.
int isLocalGrad;  // Local flag for derivative calculations.
// storage for partial derivatives w.r.t. local BOP parameters.

MatDoub MatA;     // Row matrix to store partial derivatives w.r.t. local BOP parameters
		  // whose size must be 1 x MAX_HB_PARAM.
MatDoub MatB;     // Matrix to store partial derivatives w.r.t. weights and biases of NN.
		  // whose size must be (MAX_HB_PARAM x # of fitting parameters).
// --------------------------------------------------------------------------------------

//
// ------------------------ Potential scheme ------------------------
//
int PotentialType; // Flag to choose potential type - 0: BOP 1: ANN 2: PINN
int GiMethod;      // Method to compute Gis:
		   // 0: regular Gis.
		   // 1: natural logarithm of regular Gis shifted by REF_GI.
		   // 3: for PINN, cutoff = 1.5 * Rc and no log() to Gis.
		   // 4: for PINN, cutoff = 1.5 * Rc and log() to Gis.
		   // Note: all Gis can be shifted by REF_GI irrespective of GiMethod.
int nChemSort;     // Number of chemical sorts in the system.
double REF_GI;     // Reference level to all Gis.
// ----------------------------------------------------------

//
// --------------- Neural network related global variables -------------------------
//
int nLayers;      // Number of layers including input.
int NNetInit;     // Flag to initialize weights and biases.
double MAX_RANGE; // Range of weights and biases if needed to initialize.
Layer *layer;     // Array to store layer information.
int nNNPARAM;     // Total number of fitting parameters (weights and biases).
char NNETParamFileIn[FILENAMESIZE];  // Input NN file.
char NNETParamFileOut[FILENAMESIZE]; // Output NN file.
int LogisticFuncType;  // Flag to choose logistic function:
		       // 0:sigmoid.
		       // 1:sigmoid - 0.5; i.e. 1/2 tanh(x/2).
long int rnd_seed;     // Random seed to generate weights and biases of NN.
		       // If zero, use current time as random seed.
int nSigmas;           // Number of Gaussian positions.
double *Sigmas;        // Array of size nSigmas to store positions of Gaussians (r0s).
double SS;             // Width of gaussian function used in Gi calculations.
int noGi;                  // Flag to compute Gis or read from the file.
char GiFile[FILENAMESIZE]; // File for Gis.
LSPARAMETER *__GiList;     // Array of structures to store Gis of all atoms.

int *LegendrePolyOrders;   // Array ot store orders of Legendre Polynomial for LSP:
int nLPOrders;             // Number of Legendre polynomial orders.
// ----------------------------------------------------------------------------------


double *local_eng_sum;      // Array to store locally sum of atomic energies of structures.
double *global_eng_sum;     // Array to store globally sum of atomic energies of structures.
double *local_max_pi;       // Array to store hybrid BOP parameters locally.
double *global_max_pi;      // Array to store hybrid BOP parameters of all atoms.
int max_num_atoms_per_proc; // Maximum number of atoms in each cpu.
int *num_atoms_per_proc;    // Array to store actual number of atoms in each cpu.


char element[4][3];         // Array to store chemical symbols.
double xmass[4];            // Masses of chemical sorts.
double acc;  // Accuracy of some calculations.
double Rc;   // Cutoff distance.
double Hc;   // Cutoff range.

int nelast;  // Number of points in calculations of elastic moduli.


//
// -------------- Variables for crystal structures ---------------------------
//
int max1;             // Number of times to replicate along x-direction.
int max2;             // Number of times to replicate along y-direction.
int max3;             // Number of times to replicate along z-direction.
Struc_Data *Lattice;  // Array of structures to store training data.
Struc_Data *TestSet;  // Array of structures to store testing data.
Atom *atoms;          // Array of structures to store information of each atom.
int NAtoms;           // Number of atoms in each cpu.
int nStruc;           // Number of configurations in the training data.
char datafile[FILENAMESIZE];  // Training data file.
char testfile[FILENAMESIZE];  // Test data file.
double shift_E0;      // Offset to DFT energies per atom.
int Ncluster;         // Number of clusters in the training data.
int Ntotal_bases;     // Number of atoms in the training data.
int nTestSize;        // Number of
int nTestBases;       // Number of atoms in the test data.
int nTestCluster;     // Number of clusters in the test data.
double trans0[3][3];  // Translation vectors of builtin crystal structures.
double **basis0;      // Basis vectors of builtin crystal structures.
double lc_a0;         // Equilibrium lattice constants:
double lc_b0;
double lc_c0;
char basic_struc[256]; // Name of basic structure.
double E0;             // Equilibrium energy per atom of the basic structure.
double omega0;         // Eqilibrium volume per atom of the basic structure.
// ----------------------------------------------------------------------------

//
// --------------------- BOP or Neural Network parameters ---------------------
//
VecDoub ParamVec;      // Vector to store fitting parameters;
		       // straight BOP or NN parameters.
VecDoub StepVec;       // Corresponding step sizes used only in specific algorithms.
int Nstg;              // Number of fitting stages/temperatures.
int iter0;             // Number of iterations in each stage/temp;
		       // interval to report progress.
double fTol;           // Function tolerance.
double gTol;           // Gradient tolerances.
double Tini;           // Initial temperature for simulated annealing.
double Tfin;           // Final temperature for simulated annealing.
// For temporary fixes.
double BOPconstraint;  // Regularization for straight BOP.
double NNconstraint;   // Regularization for NN parameters.
double HBconstraint;   // Regularization for local BOP.
double HBconstraint2;  // Regularization for local BOP type 2.
double CONST_BOP;      // Penalty for regularization of straight BOP parameters.
double CONST_NN;       // Penalty for weights and biases.
double CONST_HB;       // Penalty for local BOP parameters.
double CONST_HB2;      // Penalty for variation within each parameter
		       // of all atoms of NN fitted BOP parameters.
// ----------------------------------------------------------------------------


size_t funk_calls_count;

double *__BOP_EST; // Estimates for local BO parameters to be read from command file.
int use_filter;    // Flag to use a filter to local BO parameters.

//
// ----------------- Genetic algorithm related variables ----------------
//
int __optcross; // Method of crossing parents:
		// 0: swap parameters.
		// 1: weighted average.
int __ngwrite;  // Interval to write statistics of the generation.
int __np;     // Number of population to start with.
int __nf;     // Number of fittest population.
int __ng;     // Number of generation.
int __nmustg; // Number of mutation stages.
double *__s0;  // Array to store mutation size.
int *__mustep; // Array to store mutation steps.
// ----------------------------------------------------------------------

int __OptimizationMethod; // Optimization method - (0:DFP 1:GA)

int count_words(const char *line)
{

  if (!line) return 0;

  int n = strlen(line)+1;
  char *copy = NULL;
  copy = grow(copy,n,"count_words():copy");
  strcpy(copy,line);

  char *ptr;
  if ((ptr = strchr(copy,'#'))) *ptr = '\0';

  if (strtok(copy," \t\n\r\f") == NULL) {
    free(copy);
    return 0;
  }

  n = 1;

  while (strtok(NULL," \t\n\r\f")) n++;

  free(copy);
  return n;
}

void freemem()
{
  if (Sigmas) delete [] Sigmas;

  if (local_eng_sum) delete [] local_eng_sum;
  if (global_eng_sum) delete [] global_eng_sum;
  if (local_max_pi) delete [] local_max_pi;
  if (global_max_pi) delete [] global_max_pi;
  if (num_atoms_per_proc) delete [] num_atoms_per_proc;
  destroy(basis0);

  if (local_sum_partial_deriv) delete [] local_sum_partial_deriv;
  if (global_sum_partial_deriv) delete [] global_sum_partial_deriv;
  if (local_sum_hbconst_partial_deriv) delete [] local_sum_hbconst_partial_deriv;
  if (global_sum_hbconst_partial_deriv) delete [] global_sum_hbconst_partial_deriv;
  if (local_partial_deriv) delete [] local_partial_deriv;
  if (global_partial_deriv) delete [] global_partial_deriv;

  if (Lattice) {
    for (int i=0; i<nStruc; i++) {
      destroy(Lattice[i].bases);
      destroy(Lattice[i].csort);
      sfree(Lattice[i].neighbors);
      sfree(Lattice[i].nneighbors);
      if (Lattice[i].G_local) sfree(Lattice[i].G_local);
    }
    sfree(Lattice);
  }

  if (TestSet) {
    for (int i=0; i<nTestSize; i++) {
      destroy(TestSet[i].bases);
      destroy(TestSet[i].csort);
      sfree(TestSet[i].neighbors);
      sfree(TestSet[i].nneighbors);
      if (TestSet[i].G_local) sfree(TestSet[i].G_local);
    }
    sfree(TestSet);
  }

  if (layer) delete [] layer;

  if (atoms) {
    for (int i=0; i<NAtoms; i++) {
      sfree(atoms[i].nlist);
      if (atoms[i].Gi_list) sfree(atoms[i].Gi_list);
    }
    sfree(atoms);
  }

  if (__GiList) delete [] __GiList;

  if (PotentialType == 2 && use_filter) delete [] __BOP_EST;

  if (__s0) delete [] __s0;
  if (__mustep) delete [] __mustep;
}
