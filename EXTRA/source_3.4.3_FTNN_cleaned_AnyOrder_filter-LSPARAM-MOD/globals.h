#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdio.h>
#include <vector>
#include <string.h>
#include <mpi.h>
#include "mem.h"
#include "defs_consts.h"
#include "nrmatrix.h"
#include "nrvector.h"

typedef struct {
  int atomid;
  int gid, nn;
  double *nlist;
  double *Gi_list;
} Atom;

typedef struct {
  int atomid;
  int gid;
  int type;
  double *gilist;
} LSPARAMETER;

struct list3D {
  double x,y,z;
  std::vector<struct Point3D> neighbor;
};

typedef struct {
  int nnodes;
  double *Weights; // a matrix to store weights.
  double *Biases; // a vector to store biases.
  double *mvprod; // a vector to store product of a matrix and a vector.
  double *vsum; // a vector to store sum of vectors; t_mu;
  double *fdot; // a vector to store derivatives of logistic function w.r.t. their arguments.
  double *SMat; // a matrix;
} Layer;

typedef struct {
  char name[256];
  double E0;
  double E;
  double w;
  int nbas;
  int cid;
  double latvec[3][3];
  double omega0; // atomic volume
  double **bases;
  char **csort;
  int *nneighbors;
  int total_nneighbors;
  double *neighbors; // r,x,y,z sequence; 4*total_nneighbors elements;
  // local structural parameters of bases;
  // nSigma parameters for each Gi;
  // 5 Gis for each basis;
  // nSigmas*5*nbas elements;
  // G0(s[0])   G0(s[1])     .... G0(s[nSigma-1])
  // [0]        [1]          .... [nSigma-1]
  // G1(s[0])   G1(s[1])     .... G1(s[2*nSigma-1])
  // [nSigma]   [nSigma+1]   .... [2*nSigma-1]
  // G2(s[0])   G2(s[1])     .... G2(s[3*nSigma-1])
  // [2*nSigma] [2*nSigma+1] .... [3*nSigma-1]
  // G3(s[0])   G3(s[1])     .... G3(s[4*nSigma-1])
  // [3*nSigma] [3*nSigma+1] .... [4*nSigma-1]
  // G4(s[0])   G4(s[1])     .... G4(s[5*nSigmas-1])
  // [4*nSigma] [4*nSigma+1] .... [5*nSigma-1]
  // blocks of nSigma*5 values for each basis;
  double *G_local;
} Struc_Data;

enum {big_a, alpha, big_b, beta, small_h, sigma, small_a, lambda, hc};

// Global log file
extern FILE* logout;

// Global errorcode;
extern int errcode;

// MPI related
extern int me, nprocs;
extern MPI_Comm world;
extern double *local_eng_sum; // array to store locally partial energies of structures.
extern double *global_eng_sum; // array to store total energies of structures reduced globally.
extern double *local_max_pi;
extern double *global_max_pi;
extern int max_num_atoms_per_proc;
// array to store number of atoms in each processor.
extern int *num_atoms_per_proc;
// Array to store partial derivatives of local BOP
// parameters w.r.t. fitting parameters for all atoms in
// local domain.
extern double *local_partial_deriv;
// Array to store partial derivatives of local BOP
// parameters w.r.t. fitting parameters for all atoms.
extern double *global_partial_deriv;
extern double *local_sum_partial_deriv;
extern double *global_sum_partial_deriv;
// Partial derivatives related constraining local BOP parameters.
extern double *local_sum_hbconst_partial_deriv;
extern double *global_sum_hbconst_partial_deriv;

// Potential scheme
extern int PotentialType;
extern int GiMethod;
extern int nChemSort; // number of chemical sorts in the system.
extern double REF_GI; // reference level to all Gis.

// NNnets related global variables
extern int nLayers, NNetInit;
extern double MAX_RANGE;
extern Layer *layer;
extern int nNNPARAM;
extern char NNETParamFileIn[FILENAMESIZE];
extern char NNETParamFileOut[FILENAMESIZE];
// flag for choosing finite differeces (0) or
// analytical derivatives (non-zero);
// only relevant to master.
extern int isGlobalGrad;
// local flag for derivative calculations.
extern int isLocalGrad;
extern int LogisticFuncType; // option for logistic function to choose:
			     // 0:sigmoid.
			     // 1:sigmoid - 0.5; i.e. 1/2 tanh(x/2).
extern long int rnd_seed; // random seed to generate weights and biases of NN.

extern char element[4][3];
extern double xmass[4];
extern double acc, Rc, Hc;


/* variables for crystal structures */
extern int max1, max2, max3;
extern Struc_Data *Lattice;
extern Struc_Data *TestSet;
extern Atom *atoms;
extern int NAtoms;
extern int nStruc; /* number of structures in the database */
extern char datafile[FILENAMESIZE];
extern char testfile[FILENAMESIZE];
extern double shift_E0; // shift all DFT energies per atom by this ammount.
extern int Ncluster; // total number of clusters.
extern int nSigmas;
extern double *Sigmas;
extern int Ntotal_bases;
extern int nTestSize, nTestBases, nTestCluster;
extern double SS; // width of gaussian function used in local structural parameters.
extern double trans0[3][3], **basis0; // translation and basis vectors of in-built crystal structures.
extern double lc_a0, lc_b0, lc_c0; // equilibrium lattice constants.
extern char basic_struc[256];
extern double E0; // equil. energy per atom of the basic structure.
extern double omega0; // euqil. volume per atom of the basic structure.
//extern double Re0; // first neighbor distance.

// elastic properties.
extern int nelast;

/* variables for fitting parameters:
 either BOP or Neural Network parameters
*/
extern VecDoub ParamVec, StepVec;
extern int Nstg, iter0;   /* number of stages/temperatures and number of iterations in each stage/temp. */
extern double fTol, gTol; /* function and gradient tolerances */
extern double Tini, Tfin; // initial and final temperatures for simulated annealing

// For temporary fix.
extern double BOPconstraint; // constraint for signs and max. value of straight BOP parameters.
extern double NNconstraint; // constraint for NN weights and biases.
extern double HBconstraint; // constraint for signs and max. value of NN fitted BOP parameters.
extern double HBconstraint2; // constraint for variation of NN fitted BOP parameters.
extern double CONST_BOP; // for max. value of straight BOP parameters.
extern double CONST_NN; // for NN weights and biases.
extern double CONST_HB; // for NN fitted BOP parameters.
extern double CONST_HB2; // for variation within each parameter of all atoms of
			// NN fitted BOP parameters.

// flags for writing certain data.
extern int noGi; // flag to compute Gis or read from the file.
extern char GiFile[FILENAMESIZE]; // read saved Gis from this file.

extern size_t funk_calls_count;

// storage for partial derivatives w.r.t. local BOP parameters.
// size of the matrix must be 1 x MAX_HB_PARAM.
extern MatDoub MatA;
// storage for partial derivatives w.r.t. weights and biases of NN.
// size of the matrix must be MAX_HB_PARAM x number of fitting parameters.
extern MatDoub MatB;

// Array of structures to store Gis of all atoms only for collecting and writing purpose.
extern LSPARAMETER *__GiList;

// Legendre Polynomial orders for LSP.
extern int *LegendrePolyOrders;
// number of orders
extern int nLPOrders;

// Estimates for local BO parameters to be read from command file.
extern double *__BOP_EST;
extern int use_filter; // Flag to use a filter to local BO parameters.

//
// ----------------- Genetic algorithm related variables ----------------
//
extern int __optcross; // Method of crossing parents.
extern int __ngwrite;  // Interval to write statistics of the generation.
extern int __np;     // Number of population to start with.
extern int __nf;     // Number of fittest population.
extern int __ng;     // Number of generation.
extern int __nmustg; // Number of mutation stages.
extern double *__s0;  // Array to store mutation size.
extern int *__mustep; // Array to store mutation steps.
// ----------------------------------------------------------------------

extern int __OptimizationMethod; // Optimization method - (0:DFP 1:GA)

// ======================
// function declarations
// ======================

void freemem();

int count_words(const char *line);

/* random number generator function */
double ran();

int count_words(const char *line);

void sort_ascen (double *x,double *y, const int n);

//void NewSort(double *x,double **y,const int n,const int m,const int dir);

int CheckXPoints(const double *x,const int n);

std::vector<int>::iterator
locate(const double x,const double *p,const int n,std::vector<int> vec,
       std::vector<int>::iterator il,std::vector<int>::iterator ih,int dir);

void sort_ascen(double *dev,double **ppop,int npop,int nparam);

#endif // GLOBALS_H
