#ifndef WRITE_H
#define WRITE_H

#include "globals.h"

void WriteNNetParam(FILE *fout);
void WriteBOPParam(FILE *fout);
void WriteMatrixFormat(FILE *fout);
void WriteProperty();
void WriteLSP();
void WriteLSP(const int ifbop);
void WriteLSP(const Struc_Data *data, const int ns);
//void WriteDump(const char *strucname);
void WriteMaxMinDist(const char *fname);
void WriteDFTEnergy(const char *fname);
void WriteConfigs(const char *str_struct, char *filename);
//void WriteSelectEngConfigs(const char *filename, double eng_per_atom);
//void WriteDump(const char *fname, double trans[][3], double **basis, const int nb, const int tstep);
void WriteConfigs(const char *strucname, const char *filename, double trans[][3],double **basis, const int nb);
void WriteHiddenLayerParam(FILE *fout);
void WriteLastLayerOutputs(FILE *fout);
void WriteHBparam();
void compute_hMat(const double lvec[][3], double hMat[][3]);
void compute_hinvMat(const double hMat[][3], double hInvMat[][3]);
void put_atom_inside_box(double hMat[][3], double hInvMat[][3], double &x, double &y, double &z);
void WriteMatrix(FILE *f, const double *mat, const int m, const int n);

void WriteLSP_mpi();

void CartesianToDirect(const MatDoub &H, // IN: matrix of lattice vectors in row format.
		       const VecDoub &X, // IN: position in cartesian/direct space.
		       VecDoub &S,       // OUT: position in direct/cartesian space.
		       const int dir);  // IN: flag to converstion direction.

#endif // WRITE_H
