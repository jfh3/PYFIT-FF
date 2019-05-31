#ifndef NNETINTERFACE_H
#define NNETINTERFACE_H

#include "mkl.h"
#include "nrmatrix.h"
#include "globals.h"

void PartitionNNetParams (VecDoub &p);
void ReadNNetParams(const char *file, VecDoub &p,VecDoub &dy, double &s_total);
double NNET_Eng(const double *lsparam, const int nbas);
double NNET_Eng(const double *lsparam, const int Struc_Id, const int nbas);
double NNET_Eng(const double *lsparam, const int *nn, const double *nlist, const int nbas);
double iNNET_Eng(const double *lsparam, const double *nlist, const int nn);
void NNetOutput(double &f0, const int m);
void NNetOutput(const int *nn, const double *nlist,
		const int Basis_Id, double &f0, const int m);
double iNNET_Eng(const double *lsparam, const double *nlist,
		 const int nn, VecDoub &pvec);
void evaluate_nnet();

#endif // NNETINTERFACE_H
