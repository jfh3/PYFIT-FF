#ifndef ENERGY_H
#define ENERGY_H

#include "globals.h"

double Energy(const int *nn,const double *nlist,const int nbas,const VecDoub bop);

double iEnergy(const int nn,const double *nlist,const VecDoub &bop);
double Atomic_Eng(const int *nn,const double *nlist,const int Basis_Id,const VecDoub bop);
double VolumePerAtom(const double lvec[][3], const int nb);
void DFP_minimize();
void ComputeLocalStrucParam();
void iCompute_LSP(const int *nn,
		  const double *nlist,
		  const int nb,
		  double *gi);
void One_NeighborList(const double lvec[][3],
		      double **bases,
		      const int nbas,
		      int *&nneighbors,
		      double *&neighbors);
void EOS(const char *strucname, const double a, const double b, const double c);
double Funk(VecDoub &pin);
double max_filter(const VecDoub &a, const double c);
double compute_error(Struc_Data *&data, const int size, VecDoub &pin);
double compute_hb_constraint2();
//double compute_hb_constraint2(VecDoub &pin);
void ComputeLocalStrucParam(Struc_Data *&data, const int nSet);
//void icompute_hb_constraint2(VecDoub &pin);
//double icompute_hb_constraint2(const int strucId, VecDoub &pin);
double mean_sqr(const VecDoub &a, const double c);
double crystal_eng(const double trans[][3], double **basis, const int nb);
double crystalFunk(const double x);
double crystal_eng(const double trans[][3], double **basis, const int nb, const char *struc);
double specific_bop_constraint(VecDoub &pin);

void compute_LSP(); // calls compute_atomic_LSP().
void compute_atomic_LSP(const int nn,
			const double *nlist,
			double *gi);

double Area(const VecDoub a, const VecDoub b, VecDoub &axb);

#endif // ENERGY_H
