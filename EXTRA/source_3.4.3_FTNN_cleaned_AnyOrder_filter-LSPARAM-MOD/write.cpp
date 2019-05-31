#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <math.h>

#include "globals.h"
#include "write.h"
#include "NNetInterface.h"
#include "util.h"
#include "defs_consts.h"
#include "compute.h"

void WriteMaxMinDist(const char *fname)
{
  FILE *out;

  out = fopen(fname,"w");
  if (!out) {
    char buf[256];
    sprintf(buf,"cannot open %s",fname);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  // Find max and min of neighbor distances in each configuration
  // and output to the given file.
  for (int i=0; i<nStruc; i++) {
    fprintf(out,"%s",Lattice[i].name);
    double max_r, min_r;
    int k = 0;
    for (int j=0; j<Lattice[i].nbas; j++) {
      if (j) k += 4*Lattice[i].nneighbors[j-1];
      for (int n=0; n<Lattice[i].nneighbors[j]; n++) {
	if (j == 0) max_r = min_r = Lattice[i].neighbors[n*4+k];
	else {
	  max_r = MAX(max_r,Lattice[i].neighbors[n*4+k]);
	  min_r = MIN(min_r,Lattice[i].neighbors[n*4+k]);
	}
      }
    }
    fprintf(out," %f %f\n",max_r,min_r);
  }
}

void compute_hMat(const double lvec[][3], double hMat[][3])
{
  double a, b, c;

  a = sqrt(pow(lvec[0][0], 2) + pow(lvec[0][1], 2) + pow(lvec[0][2], 2));
  b = sqrt(pow(lvec[1][0], 2) + pow(lvec[1][1], 2) + pow(lvec[1][2], 2));
  c = sqrt(pow(lvec[2][0], 2) + pow(lvec[2][1], 2) + pow(lvec[2][2], 2));

  double cos_alpha, cos_beta, cos_gamma, sin_gamma;

  cos_alpha = (lvec[1][0]*lvec[2][0]+lvec[1][1]*lvec[2][1]+lvec[1][2]*lvec[2][2])/(b*c);
  cos_beta = (lvec[0][0]*lvec[2][0]+lvec[0][1]*lvec[2][1]+lvec[0][2]*lvec[2][2])/(a*c);
  cos_gamma = (lvec[0][0]*lvec[1][0]+lvec[0][1]*lvec[1][1]+lvec[0][2]*lvec[1][2])/(a*b);
  sin_gamma = sqrt(1.0 - cos_gamma*cos_gamma);

  double zeta = (cos_alpha - cos_gamma*cos_beta)/sin_gamma;

  hMat[0][0] = a;
  hMat[0][1] = b * cos_gamma;
  hMat[0][2] = c * cos_beta;
  hMat[1][0] = 0.0;
  hMat[1][1] = b * sin_gamma;
  hMat[1][2] = c * zeta;
  hMat[2][0] = 0.0;
  hMat[2][1] = 0.0;
  hMat[2][2] = c * sqrt(1.0 - cos_beta * cos_beta - zeta * zeta);

  // only for debugging purpose.
  //for (int i=0; i<3; i++) printf(" %e %e %e\n",hMat[i][0],hMat[i][1],hMat[i][2]);
}

void compute_hinvMat(const double hMat[][3], double hInvMat[][3])
{
  // compute a determinant of 3x3 matrix.
  double t1 = hMat[0][0]*(hMat[1][1]*hMat[2][2] - hMat[2][1]*hMat[1][2]);
  double t2 = hMat[0][1]*(hMat[2][0]*hMat[1][2] - hMat[1][0]*hMat[2][2]);
  double t3 = hMat[0][2]*(hMat[1][0]*hMat[2][1] - hMat[2][0]*hMat[1][1]);
  double det_h = t1 + t2 + t3;

  //printf(" determinant of h = %e\n",det_h);

  // transpose hMat and save in temporary 2d array.
  double a[3][3];
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) a[i][j] = hMat[j][i];
  }

  hInvMat[0][0] = (a[1][1]*a[2][2]-a[2][1]*a[1][2])/det_h;
  hInvMat[0][1] = (a[0][2]*a[2][1]-a[0][1]*a[2][2])/det_h;
  hInvMat[0][2] = (a[0][1]*a[1][2]-a[1][1]*a[0][2])/det_h;

  hInvMat[1][0] = (a[1][2]*a[2][0]-a[1][0]*a[2][2])/det_h;
  hInvMat[1][1] = (a[0][0]*a[2][2]-a[0][2]*a[2][0])/det_h;
  hInvMat[1][2] = (a[0][2]*a[1][0]-a[0][0]*a[1][2])/det_h;

  hInvMat[2][0] = (a[1][0]*a[2][1]-a[1][1]*a[2][0])/det_h;
  hInvMat[2][1] = (a[0][1]*a[2][0]-a[0][0]*a[2][1])/det_h;
  hInvMat[2][2] = (a[0][0]*a[1][1]-a[0][1]*a[1][0])/det_h;

  // only for debugging purpose.
  //for (int i=0; i<3; i++) printf(" %e %e %e\n",hInvMat[i][0],hInvMat[i][1],hInvMat[i][2]);
  //exit(0);
}

void CartesianToDirect(const MatDoub &H, // IN: matrix of lattice vectors in row format.
		       const VecDoub &X, // IN: position in cartesian/direct space.
		       VecDoub &S,       // OUT: position in direct/cartesian space.
		       const int dir)   // IN: flag to converstion direction.
{
  // This subroutine may be used to wrap atoms:
  // invH * X = S
  // S' = S - floor(S)
  // X' = H * S'

  if (H.ncols () != 3 && H.nrows () != 3) {
    printf("Error: cannot do conversion .... [%s:%d]\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  if (X.size () != 3 && S.size () != 3) {
    printf("Error: cannot do conversion .... [%s:%d]\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  MatDoub tmp(X.size (), 1);

  // Copy X to tmp.
  for (size_t i=0; i<X.size (); i++) tmp[i][0] = X[i];

  // Direction of converstion.
  if (dir == 0) {
    // Compute inverse of H.
    MatDoub invH(H.inv ());
    // Multiply invH and tmp.
    MatDoub SS(invH * tmp);
    // Copy SS to S
    for (size_t i=0; i<S.size (); i++) S[i] = SS[i][0] - floor(SS[i][0]);
  } else if (dir == 1) {
    // Copy H to HH; just to fix 'const' qualifier requirement of matrix multiplication, '*', by a compiler !!!
    MatDoub HH(H);
    // Multiply H and tmp.
    MatDoub XX(HH * tmp);
    // Copy SS to S
    for (size_t i=0; i<S.size (); i++) S[i] = XX[i][0];
  }
}

void put_atom_inside_box(double hMat[][3], double hInvMat[][3], double &x, double &y, double &z)
{
  double s[3], ss[3];

  s[0] = hInvMat[0][0] * x + hInvMat[0][1] * y + hInvMat[0][2] * z;
  s[1] = hInvMat[1][0] * x + hInvMat[1][1] * y + hInvMat[1][2] * z;
  s[2] = hInvMat[2][0] * x + hInvMat[2][1] * y + hInvMat[2][2] * z;
  //printf(" %d %e %e %e\n",i+1,s[0],s[1],s[2]);
  ss[0] = s[0] - floor(s[0]);
  ss[1] = s[1] - floor(s[1]);
  ss[2] = s[2] - floor(s[2]);
  //printf(" %d %e %e %e\n",i+1,ss[0],ss[1],ss[2]);
  x = hMat[0][0] * ss[0] + hMat[0][1] * ss[1] + hMat[0][2] * ss[2];
  y = hMat[1][0] * ss[0] + hMat[1][1] * ss[1] + hMat[1][2] * ss[2];
  z = hMat[2][0] * ss[0] + hMat[2][1] * ss[1] + hMat[2][2] * ss[2];
}


/*void WriteDump(const char *strucname)
{
  int ncount = 0;
  FILE *out;
  char buf[256];

  sprintf(buf,"%s.dump",strucname);
  out = fopen(buf,"w");
  if (!out) {
    sprintf(buf,"cannot open %s.dump",strucname);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  double HMat[3][3], HInvMat[3][3];

  for (int i=0; i<nStruc; ++i) {
    if (strcmp(Lattice[i].name,strucname) == 0) {
      fprintf(out,"ITEM: TIMESTEP\n");
      fprintf(out,"%d\n",ncount++);
      fprintf(out,"ITEM: NUMBER OF ATOMS\n");
      fprintf(out,"%d\n",Lattice[i].nbas);
      fprintf(out,"ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
      //if (Lattice[i].latvec[1][0] < 0.0 || Lattice[i].latvec[2][0] < 0.0 || Lattice[i].latvec[2][1] < 0.0) {
	//printf(" bx, cx, and cy must be non-negative ....\n");
	//fclose(out);
	//return;
      //}
      //double xhi = Lattice[i].latvec[0][0] + MAX(MAX(0.0,Lattice[i].latvec[1][0]),MAX(Lattice[i].latvec[2][0],Lattice[i].latvec[1][0]+Lattice[i].latvec[2][0]));
      //double yhi = Lattice[i].latvec[1][1] + MAX(0.0,Lattice[i].latvec[2][1]);
      //fprintf(out,"%f %f %f\n",0.0,xhi,Lattice[i].latvec[1][0]);
      //fprintf(out,"%f %f %f\n",0.0,yhi,Lattice[i].latvec[2][0]);
      //fprintf(out,"%f %f %f\n",0.0,Lattice[i].latvec[2][2],Lattice[i].latvec[2][1]);

      compute_hMat(Lattice[i].latvec,HMat);
      compute_hinvMat(Lattice[i].latvec,HInvMat);
      double xlo = 0.0 + MIN(MIN(0.0,HMat[0][1]),MIN(HMat[0][2],HMat[0][1]+HMat[0][2]));
      double xhi = HMat[0][0] + MAX(MAX(0.0,HMat[0][1]),MAX(HMat[0][2],HMat[0][1]+HMat[0][2]));
      double ylo = 0.0 + MIN(0.0,HMat[1][2]);
      double yhi = HMat[1][1] + MAX(0.0,HMat[1][2]);
      fprintf(out,"%f %f %f\n",xlo,xhi,HMat[0][1]);
      fprintf(out,"%f %f %f\n",ylo,yhi,HMat[0][2]);
      fprintf(out,"%f %f %f\n",0.0,HMat[2][2],HMat[1][2]);

      fprintf(out,"ITEM: ATOMS id type x y z\n");
      for (int j=0; j<Lattice[i].nbas; ++j) {
	put_atom_inside_box(HMat,HInvMat,Lattice[i].bases[j][0],Lattice[i].bases[j][1],Lattice[i].bases[j][2]);
	fprintf(out,"%d %d %f %f %f\n",j+1,1,Lattice[i].bases[j][0],
	    Lattice[i].bases[j][1],Lattice[i].bases[j][2]);
      }
						//printf("H matrix:\n");
      //for (int i=0; i<3; i++) {
	//printf("%e %e %e\n",HMat[i][0],HMat[i][1],HMat[i][2]);
      //}
    }
  }

  fclose(out);
}
*/

/*
void WriteLSP()
{
  int size;

  size = layer[0].nnodes;

  if (me == 0) {
    FILE *out, *out2;

    out = fopen("LSParam.dat","w");

    if (!out) {
      printf("Error: cannot open LSParam.dat .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_NO_SUCH_FILE);
      exit(EXIT_FAILURE);
    }

    out2 = fopen("Gis_dist.dat","w");

    if (!out) {
      printf("Error: cannot open Gis_dist.dat .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_NO_SUCH_FILE);
      exit(EXIT_FAILURE);
    }

    if (__GiList) {
      for (int i=0; i<Ntotal_bases; i++) {
	fprintf(out, " %d %d %s", __GiList[i].atomid, __GiList[i].gid,
		Lattice[__GiList[i].gid].name);
	fprintf(out2, " %d %d %s", __GiList[i].atomid, __GiList[i].gid,
		Lattice[__GiList[i].gid].name);
	double max_val, min_val;
	double sum = 0.0;
	double sum2 = 0.0;
	for (int k=0; k<size; ++k) {
	  double tmp = __GiList[i].gilist[k];
	  fprintf(out, " %e", tmp);
	  if (!k) max_val = min_val = tmp;
	  else {
	    max_val = MAX(max_val,tmp);
	    min_val = MIN(min_val,tmp);
	  }
	  sum += tmp;
	  sum2 += pow(tmp, 2);
	}
	sum /= (size - 2);
	fprintf(out2," %e %e %e %e\n", sum, sqrt(sum2/(size - 2) - sum*sum), max_val, min_val);
	fprintf(out,"\n");
      }
      fclose(out);
      fclose(out2);
    } else {
      fclose(out);
      fclose(out2);
      printf("Error: __GiList is empty .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_OTHER);
      exit(0);
    }
  }
}
*/

/*
void WriteLSP(const int ifbop)
{
  FILE *out, *out2;
  char fname[1024];


  if (ifbop == 0) sprintf(fname, "LSParam.dat");
  else if (ifbop == 1) sprintf(fname, "LSParam_wBOP.dat");

  out = fopen(fname, "w");

  if (!out) {
    printf("Error: cannot open file:%s .... [%s:%d]\n", fname, __FILE__, __LINE__);
    MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  out2 = fopen("Gis_dist.dat","w");

  if (!out) {
    errmsg("cannot open Gis_dist.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  int size = nSigmas*MAX_LSP;

  for (int i=0; i<nStruc; i++) {
    int block_shift = 0;
    for (int j=0; j<Lattice[i].nbas; ++j) {
      fprintf(out," %s %d",Lattice[i].name,Lattice[i].cid);
      fprintf(out2," %s %d",Lattice[i].name,Lattice[i].cid);
      double max_val, min_val;
      double sum = 0.0;
      double sum2 = 0.0;
      if (j) block_shift += size;
      for (int k=0; k<size; ++k) {
	double tmp = Lattice[i].G_local[k+block_shift];
	if (!k) max_val = min_val = tmp;
	else {
	  max_val = MAX(max_val,tmp);
	  min_val = MIN(min_val,tmp);
	}
	sum += Lattice[i].G_local[k+block_shift];
	sum2 += pow(Lattice[i].G_local[k+block_shift],2);
	fprintf(out," %e",Lattice[i].G_local[k+block_shift]);
      }
      sum /= size;
      fprintf(out2," %e %e %e\n",sum,sqrt(sum2/size - sum*sum),fabs(max_val - min_val));

      // write local BOP parameters of each basis atom.
      if (ifbop) {
	// update vsum of layer[0]
	for (int l=0; l<size; l++) layer[0].vsum[l] = Lattice[i].G_local[l + block_shift];
	evaluate_nnet();
	int n = layer[nLayers-1].nnodes;
	if (n != MAX_HB_PARAM) {
	  printf("Error: size mismatch from file %s at line %d\n", __FILE__, __LINE__);
	  if (nprocs > 1) MPI_Abort(world,errcode);
	  exit(EXIT_FAILURE);
	}
	for (int l=0; l<n; l++) fprintf(out, " %e", layer[nLayers-1].vsum[l]);
      }
      fprintf(out,"\n");
    }
  }

  fclose(out);
  fclose(out2);
}
*/
/*
void WriteLSP(const Struc_Data *data, const int ns)
{
  FILE *out, *out2;

  out = fopen("LSParam.dat","w");

  if (!out) {
    errmsg("cannot open LSParam.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  out2 = fopen("Gis_dist.dat","w");

  if (!out) {
    errmsg("cannot open Gis_dist.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  int size = nSigmas*MAX_LSP;

  for (int i=0; i<ns; i++) {
    int block_shift = 0;
    for (int j=0; j<data[i].nbas; ++j) {
      fprintf(out," %s %d",data[i].name,data[i].cid);
      fprintf(out2," %s %d",data[i].name,data[i].cid);
      double max_val, min_val;
      double sum = 0.0;
      double sum2 = 0.0;
      if (j) block_shift += size;
      for (int k=0; k<size; ++k) {
	double tmp = data[i].G_local[k+block_shift];
	if (!k) max_val = min_val = tmp;
	else {
	  max_val = MAX(max_val,tmp);
	  min_val = MIN(min_val,tmp);
	}
	fprintf(out," %e",data[i].G_local[k+block_shift]);
	sum += data[i].G_local[k+block_shift];
	sum2 += pow(data[i].G_local[k+block_shift],2);
      }
      sum /= size;
      fprintf(out2," %e %e %e\n",sum,sqrt(sum2/size - sum*sum),fabs(max_val - min_val));
      fprintf(out,"\n");
    }
  }

  fclose(out);
  fclose(out2);
}
*/

// Write DFT energies per atom along with
// names and volumes per atom.
void WriteDFTEnergy(const char *fname)
{
  FILE *out;

  out = fopen(fname,"w");

  if (!out) {
    char buf[256];
    sprintf(buf,"cannot open %s",fname);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  fprintf(out,"# All energies per atom shifted by %f\n",shift_E0);
  fprintf(out,"# S.N. Structure Cluster_id Vol_per_atom DFT_eng(eV/atom)\n");
  for (int i=0; i<nStruc; i++) {
    fprintf(out,"%d %-20s %d %17.8e %17.8e\n",i+1,Lattice[i].name,
	    Lattice[i].cid,Lattice[i].omega0,Lattice[i].E0/Lattice[i].nbas);
  }

  fclose(out);
}

// =============================
// print the computed properties
// =============================
void WriteProperty()
{
  FILE *out;

  out = fopen("E_vs_V.dat","w");

  if (!out) {
    errmsg("cannot open E_vs_V.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  fprintf(out,"# All DFT energies per atom shifted by %f\n",shift_E0);
  fprintf(out,"# Structure Cluster_id Vol_per_atom DFT_eng(eV/atom) Computed_eng(eV/atom)\n");
  for (int i=0; i<nStruc; i++) {
    fprintf(out,"%-20s %d %17.8e %17.8e %17.8e\n",Lattice[i].name,
	    Lattice[i].cid,Lattice[i].omega0,Lattice[i].E0/Lattice[i].nbas,
	    Lattice[i].E/Lattice[i].nbas);
  }
  fclose(out);

  // compute total number of clusters.
  int prev_cid;
  int nc = 0;
  for (int i=0; i<nStruc; i++) {
    if (!i) {
      prev_cid = Lattice[i].cid;
      nc++;
    } else {
      if (prev_cid != Lattice[i].cid) {
	int ifound = 0;
	for (int j=0; j<i; j++) {
	  if (Lattice[j].cid == Lattice[i].cid) ifound = 1;
	}
	if (!ifound) {
	  nc++;
	  prev_cid = Lattice[i].cid;
	}
      }
    }
  }
  printf(" Total number of clusters: %d\n",nc);
  // compute number of configurations in each cluster.
  int *ncon;
  double *dev;
  int *nnbas;
  ncon = new int [nc];
  dev = new double [nc];
  nnbas = new int [nc];
  for (int i=0; i<nc; i++) {
    ncon[i] = 0;
    dev[i] = 0.0;
    nnbas[i] = 0;
  }
  for (int i=0; i<nStruc; i++) {
    ncon[Lattice[i].cid-1]++;
    dev[Lattice[i].cid-1] += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas,2);
    nnbas[Lattice[i].cid-1] += Lattice[i].nbas;
  }

  out = fopen("RMSE_vs_cluster.dat","w");

  if (!out) {
    errmsg("cannot open RMSE_vs_cluster.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  double t_dev = 0.0;
  int t_c = 0;
  fprintf(out,"# C_Id Nconfigs Nbases RMSError(eV/atom)\n");
  for (int i=0; i<nc; i++) {
    t_dev += dev[i];
    t_c += ncon[i];
    fprintf(out,"%d %d %d %f\n",i+1,ncon[i],nnbas[i],sqrt(dev[i]/ncon[i]));
  }
  printf(" Check total number of configurations: %d\n",t_c);
  printf(" Check total RMSError: %f\n",sqrt(t_dev/t_c));
  fclose(out);

  delete [] ncon;
  delete [] dev;
  delete [] nnbas;
}


void WriteMatrixFormat(FILE *fout)
{
  PartitionNNetParams(ParamVec);
  fprintf(fout,"==== Neural network parameters in matrix format ====\n");
  for (int i=0; i<nLayers; i++) {
    if (i == 0) {
      fprintf(fout," Vector[%d]:\n",layer[i].nnodes);
      for (int j=0; j<layer[i].nnodes; j++) fprintf(fout," %f",layer[i].vsum[j]);
      fprintf(fout,"\n");
    }
    else {
      fprintf(fout," Matrix[%d x %d]:\n",layer[i-1].nnodes,layer[i].nnodes);
      for (int j=0; j<layer[i-1].nnodes; j++) {
	for (int k=0; k<layer[i].nnodes; k++) fprintf(fout," %f",layer[i].Weights[j*layer[i].nnodes+k]);
	fprintf(fout,"\n");
      }
      fprintf(fout," Vector[%d]:\n",layer[i].nnodes);
      for (int j=0; j<layer[i].nnodes; j++) fprintf(fout," %f",layer[i].Biases[j]);
      fprintf(fout,"\n");
    }
  }
}

void WriteMatrix(FILE *f, const double *mat, const int m, const int n)
{
  fprintf(f," Matrix[%d x %d]:\n",m,n);
  for (int j=0; j<m; j++) {
    for (int k=0; k<n; k++) fprintf(f," %f",mat[j*n + k]);
    fprintf(f,"\n");
  }
}

void WriteNNetParam(FILE *fout)
{
  // Gi version, reference Gi and logistic function type.
  fprintf(fout," %d %f %d - Gi version, reference Gi and type of logistic function.\n",
	  GiMethod, REF_GI, LogisticFuncType);

  // number of chemical species in the system.
  fprintf(fout," %d - number of chemical species in the system.\n", nChemSort);

  // element name and mass.
  for (int i=0; i<nChemSort; i++) {
    fprintf(fout," %s %f\n",element[i],xmass[i]);
  }

  // flag and max. range
  fprintf(fout," %d %f %f %f %f\n",NNetInit,MAX_RANGE,Rc,Hc,SS);

  // Write number of Legendre polynomials and orders.
  fprintf(fout, " %d", nLPOrders);
  for (int i=0; i<nLPOrders; i++) fprintf(fout, " %d", LegendrePolyOrders[i]);
  fprintf(fout, "\n");

  // Write number of Sigmas and values.
  fprintf(fout," %d",nSigmas);
  for (int i=0; i<nSigmas; ++i) fprintf(fout," %.2f",Sigmas[i]);
  fprintf(fout,"\n");

  // Write flag and estimates of BO parameters.
  fprintf(fout, " %d", use_filter);
  for (int i=0; i<MAX_HB_PARAM; i++) fprintf(fout, " %f", __BOP_EST[i]);
  fprintf(fout, "\n");

  // Write information of NN architecture.
  fprintf(fout," %d",nLayers);
  for (int l=0; l<nLayers; l++) {
    fprintf(fout," %d",layer[l].nnodes);
  }
  fprintf(fout,"\n");

  // write parameters: weights then biases of each layer
  for (int i=0; i<nNNPARAM; i++)
    fprintf(fout,"%16.8e %8.4f\n",ParamVec[i],StepVec[i]);
  fprintf(fout,"\n");
  fflush(fout);
}


void WriteHiddenLayerParam(FILE *fout)
{
  // Gi version, reference Gi and logistic function type.
  fprintf(fout," %d %f %d - Gi version, reference Gi and type of logistic function.\n",
	  GiMethod, REF_GI, LogisticFuncType);
  // number of chemical species in the system.
  fprintf(fout," %d - number of chemical species in the system.\n",nChemSort);
  // element name and mass.
  for (int i=0; i<nChemSort; i++) {
    fprintf(fout," %s %f\n",element[i],xmass[i]);
  }
  // flag and max. range
  fprintf(fout," %d %f %f %f %f\n",NNetInit,MAX_RANGE,Rc,Hc,SS);

  // Write number of Legendre polynomials and orders.
  fprintf(fout, " %d", nLPOrders);
  for (int i=0; i<nLPOrders; i++) fprintf(fout, " %d", LegendrePolyOrders[i]);
  fprintf(fout, "\n");

  // Write number of Sigmas and values.
  fprintf(fout," %d",nSigmas);
  for (int i=0; i<nSigmas; ++i) fprintf(fout," %.2f",Sigmas[i]);
  fprintf(fout,"\n");

  // Write flag and estimates of BO parameters.
  fprintf(fout, " %d", use_filter);
  for (int i=0; i<MAX_HB_PARAM; i++) fprintf(fout, " %f", __BOP_EST[i]);
  fprintf(fout, "\n");

  // Write information of NN architecture.
  fprintf(fout," %d",nLayers);
  for (int l=0; l<nLayers-1; l++) {
    fprintf(fout," %d",layer[l].nnodes);
  }
  fprintf(fout, " 0\n");
  fflush(fout);

  // write parameters: weights then biases of each
  // hidden layer from left to right.
  int i = 0;
  // The first layer is the input layer and skipped.
  // Node values of input layer are structural
  // properties of atoms.
  for (int l=1; l<nLayers-1; l++) {
    // weights: m x n matrix
    for (int m=0; m<layer[l-1].nnodes; m++) { // rows
      for (int n=0; n<layer[l].nnodes; n++) { // cols
	//layer[l].Weights[m*layer[l].nnodes+n] = p[i++];
	fprintf(fout,"%16.8e %8.4f\n",ParamVec[i],StepVec[i]);
	fflush(fout);
	i++;
      }
    }
    // biases: vector of size nnodes.
    for (int m=0; m<layer[l].nnodes; m++) { // cols
      //layer[l].Biases[m] = p[i++];
      fprintf(fout,"%16.8e %8.4f\n",ParamVec[i],StepVec[i]);
      fflush(fout);
      i++;
    }
  }
}


void WriteBOPParam(FILE *fout)
{
  fprintf(fout," %-10s %25.18e %f\n","A",ParamVec[big_a],StepVec[big_a]);
  fprintf(fout," %-10s %25.18e %f\n","alpha",ParamVec[alpha],StepVec[alpha]);
  fprintf(fout," %-10s %25.18e %f\n","B",ParamVec[big_b],StepVec[big_b]);
  fprintf(fout," %-10s %25.18e %f\n","beta",ParamVec[beta],StepVec[beta]);
  fprintf(fout," %-10s %25.18e %f\n","a",ParamVec[small_a],StepVec[small_a]);
  fprintf(fout," %-10s %25.18e %f\n","h",ParamVec[small_h],StepVec[small_h]);
  fprintf(fout," %-10s %25.18e %f\n","lambda",ParamVec[lambda],StepVec[lambda]);
  fprintf(fout," %-10s %25.18e %f\n","sigma",ParamVec[sigma],StepVec[sigma]);
  //fprintf(fout," %-10s %25.18e %f\n","eta",ParamVec[eta],StepVec[eta]);
  fprintf(fout," %-10s %25.18e %f\n","hc",ParamVec[hc],StepVec[hc]);
  fprintf(fout," %-10s %25.18e %f\n","rc",Rc,0.0);
}

void WriteConfigs(const char *str_struct, char *filename)
{
  FILE *fp;

  fp = fopen(filename,"w");

  for (int i=0; i<nStruc; i++) {
    if (strcmp(str_struct,Lattice[i].name) == 0) {
      fprintf(fp,"%s # comment line\n",Lattice[i].name);
      fprintf(fp,"1 # universal scaling factor\n");
      fprintf(fp,"%.8e %.8e %.8e # first Bravais lattice vector\n",
	      Lattice[i].latvec[0][0],Lattice[i].latvec[0][1],
	Lattice[i].latvec[0][2]);
      fprintf(fp,"%.8e %.8e %.8e # second Bravais lattice vector\n",
	      Lattice[i].latvec[1][0],Lattice[i].latvec[1][1],
	  Lattice[i].latvec[1][2]);
      fprintf(fp,"%.8e %.8e %.8e # third Bravais lattice vector\n",
	      Lattice[i].latvec[2][0],Lattice[i].latvec[2][1],
	  Lattice[i].latvec[2][2]);
      fprintf(fp,"%d\n",Lattice[i].nbas);
      fprintf(fp,"C # direct or cartesian, only first letter is significant\n");
      for (int j=0; j<Lattice[i].nbas; j++) {
	fprintf(fp,"%.8e %.8e %.8e\n",Lattice[i].bases[j][0],
	    Lattice[i].bases[j][1],Lattice[i].bases[j][2]);
      }
      fprintf(fp,"%.8e # total energy\n",Lattice[i].E0);
    }
  }

  fclose(fp);
}

/*void WriteSelectEngConfigs(const char *filename, double eng_per_atom)
{
  FILE *fp;

  fp = fopen(filename,"w");

  for (int i=0; i<nStruc; i++) {
    if (Lattice[i].E0/Lattice[i].nbas <= eng_per_atom) {
      fprintf(fp,"%s # comment line\n",Lattice[i].name);
      fprintf(fp,"1 # universal scaling factor\n");
      fprintf(fp,"%.8e %.8e %.8e # first Bravais lattice vector\n",
	      Lattice[i].latvec[0][0],Lattice[i].latvec[0][1],
	Lattice[i].latvec[0][2]);
      fprintf(fp,"%.8e %.8e %.8e # second Bravais lattice vector\n",
	      Lattice[i].latvec[1][0],Lattice[i].latvec[1][1],
	  Lattice[i].latvec[1][2]);
      fprintf(fp,"%.8e %.8e %.8e # third Bravais lattice vector\n",
	      Lattice[i].latvec[2][0],Lattice[i].latvec[2][1],
	  Lattice[i].latvec[2][2]);
      fprintf(fp,"%d\n",Lattice[i].nbas);
      fprintf(fp,"C # direct or cartesian, only first letter is significant\n");
      for (int j=0; j<Lattice[i].nbas; j++) {
	fprintf(fp,"%.8e %.8e %.8e\n",Lattice[i].bases[j][0],
	    Lattice[i].bases[j][1],Lattice[i].bases[j][2]);
      }
      fprintf(fp,"%.8e # total energy\n",Lattice[i].E0);
    }
  }

  fclose(fp);
}
*/

void WriteConfigs(const char *strucname, const char *filename, double trans[][3], double **basis, const int nb)
{
  FILE *fp;

  fp = fopen(filename,"w");
  fprintf(fp,"%s # comment line\n",strucname);
  fprintf(fp,"1 # universal scaling factor\n");
  fprintf(fp,"%.8e %.8e %.8e # first Bravais lattice vector\n",
	  trans[0][0],trans[0][1],trans[0][2]);
  fprintf(fp,"%.8e %.8e %.8e # second Bravais lattice vector\n",
	  trans[1][0],trans[1][1],trans[1][2]);
  fprintf(fp,"%.8e %.8e %.8e # third Bravais lattice vector\n",
	  trans[2][0],trans[2][1],trans[2][2]);
  fprintf(fp,"%d\n",nb);
  fprintf(fp,"C # direct or cartesian, only first letter is significant\n");
  for (int j=0; j<nb; j++) {
    fprintf(fp,"%.8e %.8e %.8e %s\n",basis[j][0],basis[j][1],
	basis[j][2],element[0]);
  }
  fclose(fp);
}

/*void WriteDump(const char *fname, double trans[][3], double **basis, const int nb, const int tstep)
{
  FILE *out;
  out = fopen(fname,"a+");
  fprintf(out,"ITEM: TIMESTEP\n");
  fprintf(out,"%d\n",tstep);
  fprintf(out,"ITEM: NUMBER OF ATOMS\n");
  fprintf(out,"%d\n",nb);
  fprintf(out,"ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
  //if (trans[1][0] < 0.0 || trans[2][0] < 0.0 || trans[2][1] < 0.0) {
    //printf(" bx, cx, and cy must be non-negative ....\n");
    //fclose(out);
    //return;
  //}
  //double xhi = trans[0][0] + MAX(MAX(0.0,trans[1][0]),MAX(trans[2][0],trans[1][0]+trans[2][0]));
  //double yhi = trans[1][1] + MAX(0.0,trans[2][1]);
  //fprintf(out,"%f %f %f\n",0.0,xhi,trans[1][0]);
  //fprintf(out,"%f %f %f\n",0.0,yhi,trans[2][0]);
  //fprintf(out,"%f %f %f\n",0.0,trans[2][2],trans[2][1]);

  double HMat[3][3], HInvMat[3][3];
  compute_hMat(trans,HMat);
  compute_hinvMat(trans,HInvMat);
  double xlo = 0.0 + MIN(MIN(0.0,HMat[0][1]),MIN(HMat[0][2],HMat[0][1]+HMat[0][2]));
  double xhi = HMat[0][0] + MAX(MAX(0.0,HMat[0][1]),MAX(HMat[0][2],HMat[0][1]+HMat[0][2]));
  double ylo = 0.0 + MIN(0.0,HMat[1][2]);
  double yhi = HMat[1][1] + MAX(0.0,HMat[1][2]);
  fprintf(out,"%f %f %f\n",xlo,xhi,HMat[0][1]);
  fprintf(out,"%f %f %f\n",ylo,yhi,HMat[0][2]);
  fprintf(out,"%f %f %f\n",0.0,HMat[2][2],HMat[1][2]);

  fprintf(out,"ITEM: ATOMS id type x y z\n");
  for (int i=0; i<nb; i++) {
    put_atom_inside_box(HMat,HInvMat,basis[i][0],basis[i][1],basis[i][2]);
    fprintf(out,"%d %d %f %f %f\n",i+1,1,basis[i][0],basis[i][1],basis[i][2]);
  }
  fclose(out);

		//printf("H matrix:\n");
  //for (int i=0; i<3; i++) {
				//printf("%e %e %e\n",HMat[i][0],HMat[i][1],HMat[i][2]);
  //}
}
*/

void WriteLastLayerOutputs(FILE *fout)
{
  int size;

  size = layer[0].nnodes;

  for (int i=0; i<Ntotal_bases; i++) { // loop over bases.
    fprintf(fout, "%d %d %s", __GiList[i].atomid, __GiList[i].gid, Lattice[__GiList[i].gid].name);
    // load Gis of this atom to the input layer.
    for (int j=0; j<size; j++) layer[0].vsum[j] = __GiList[i].gilist[j];
    // pass Gis through the NN.
    evaluate_nnet();
    for (int k=0; k<layer[nLayers-2].nnodes; k++) fprintf(fout," %.8e",layer[nLayers-2].vsum[k]);
    fprintf(fout,"\n");
    fflush(fout);
  }
}

void WriteHBparam()
{
  // Write hybrid BOP parameters of all atoms.
  FILE *out;
  if (nprocs > 1) {
    out = fopen("hb_BOP.dat","w");
    for (int i=0; i<nprocs; i++) { // loop over blocks.
      for (int j=0; j<num_atoms_per_proc[i]; j++) {// loop over atoms within each block.
	for (int k=0; k<MAX_HB_PARAM; k++)
	  fprintf(out, " %e", global_max_pi[i * max_num_atoms_per_proc * MAX_HB_PARAM + j * MAX_HB_PARAM + k]);
	fprintf(out,"\n");
      }
    }
    fclose(out);
  }
}

void WriteLSP()
{
  FILE *out, *out2, *out3;

  out = fopen("LSParam.dat","w");

  if (!out) {
    errmsg("cannot open LSParam.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  out2 = fopen("Gis_dist.dat","w");

  if (!out) {
    errmsg("cannot open Gis_dist.dat",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  out3 = fopen("Nbdlist.dat","w");
  if (!out3) {
    errmsg("cannot open Nbdlist",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  int NATOM;
  NATOM=0;
  for (int i=0; i<nStruc; i++) {NATOM=NATOM+Lattice[i].nbas;}


  //---------NB LIST HEADER----------
  // Gi version, reference Gi and logistic function type.
  fprintf(out3,"# %d %f %d - Gi version, reference Gi and type of logistic fucntion.\n",GiMethod,REF_GI,LogisticFuncType);
    // number of chemical species in the system.
  fprintf(out3,"# %d - number of chemical species in the system.\n",nChemSort);
  // element name and mass.
  for (int i=0; i<nChemSort; i++) {
    fprintf(out3,"# %s %f\n",element[i],xmass[i]);
  }
  // flag and max. range
  fprintf(out3,"# %d %f %f %f %f\n",NNetInit,MAX_RANGE,Rc,Hc,SS);

  // Write number of Sigmas and values.
  fprintf(out3,"# %d",nSigmas);
  for (int i=0; i<nSigmas; ++i) fprintf(out3," %.4f",Sigmas[i]);
  fprintf(out3,"\n");

  fprintf(out3,"# %d -PotentialType \n",PotentialType);
  fprintf(out3,"# %d -number of structures \n",nStruc);
  fprintf(out3,"# %d -number of atoms \n",NATOM);

  fprintf(out3,"#ATOM-ID GROUP-NAME GROUP-ID STRUCTURE-ID STRUCTURE-Natom STRUCTURE-E_DFT\n");
   ////   fprintf(out3,"ATOM-%d %s %d %d %d %e  \n",atom_count,Lattice[i].name,Lattice[i].cid,i,Lattice[i].nbas,Lattice[i].E0);
  // write header lines
 //   fprintf(out3," %d",nLayers);
  //  for (int l=0; l<nLayers; l++) {
 //     fprintf(out3," %d",layer[l].nnodes);
 //   }
 //   fprintf(out3,"\n");

  //---------LSPARAM LIST HEADER----------
  // Gi version, reference Gi and logistic function type.
  fprintf(out,"# %d %f %d - Gi version, reference Gi and type of logistic fucntion.\n",GiMethod,REF_GI,LogisticFuncType);
    // number of chemical species in the system.
  fprintf(out,"# %d - number of chemical species in the system.\n",nChemSort);
  // element name and mass.
  for (int i=0; i<nChemSort; i++) {
    fprintf(out,"# %s %f\n",element[i],xmass[i]);
  }
  // flag and max. range
  fprintf(out,"# %d %f %f %f %f\n",NNetInit,MAX_RANGE,Rc,Hc,SS);

  // Write number of Sigmas and values.
  fprintf(out,"# %d",nSigmas);
  for (int i=0; i<nSigmas; ++i) fprintf(out," %.4f",Sigmas[i]);
  fprintf(out,"\n");

  fprintf(out,"# %d -PotentialType \n",PotentialType);
  fprintf(out,"# %d -number of structures \n",nStruc);
  fprintf(out,"# %d -number of atoms \n",NATOM);

  fprintf(out,"#GROUP-NAME GROUP-ID STRUCTURE-ID STRUCTURE-Natom STRUCTURE-E_DFT Gi\n");
  ////fprintf(out," %s %d %d %d %e",Lattice[i].name,Lattice[i].cid,i,Lattice[i].nbas,Lattice[i].E0);
int atom_count;
atom_count=1;
  int size = nSigmas*MAX_LSP;
  for (int i=0; i<nStruc; i++) {
    int block_shift = 0;
    int k1 = 0;
    for (int j=0; j<Lattice[i].nbas; ++j) {
      fprintf(out," %s %d %d %d %e",Lattice[i].name,Lattice[i].cid,i,Lattice[i].nbas,Lattice[i].E0);
      fprintf(out2," %s %d",Lattice[i].name,Lattice[i].cid);
      fprintf(out3,"ATOM-%d %s %d %d %d %15.10e %e \n",atom_count,Lattice[i].name,Lattice[i].cid,i,Lattice[i].nbas,Lattice[i].E0,Lattice[i].omega0);
     fprintf(out3,"Gi ");
	atom_count=atom_count+1;
      double max_val, min_val;
      double sum = 0.0;
      double sum2 = 0.0;
      if (j) block_shift += size;
      for (int k=0; k<size; ++k) {
	double tmp = Lattice[i].G_local[k+block_shift];
	if (!k) max_val = min_val = tmp;
	else {
	  max_val = MAX(max_val,tmp);
	  min_val = MIN(min_val,tmp);
	}
	fprintf(out," %15.10e",Lattice[i].G_local[k+block_shift]);
	fprintf(out3," %15.10e",Lattice[i].G_local[k+block_shift]);
	sum += Lattice[i].G_local[k+block_shift];
	sum2 += pow(Lattice[i].G_local[k+block_shift],2);
      }
      fprintf(out3,"\n");
      fprintf(out3,"NBL %d",Lattice[i].nneighbors[j]);
      if (j) k1 += 4*Lattice[i].nneighbors[j-1];
      for (int n1=0; n1<Lattice[i].nneighbors[j]; n1++) {
		//fprintf(out3," %e %e %e %e",Lattice[i].neighbors[n*4+k+0],Lattice[i].neighbors[n*4+k+1],Lattice[i].neighbors[n*4+k+2],Lattice[i].neighbors[n*4+k+3]);
		fprintf(out3," %15.10e %15.10e %15.10e",Lattice[i].neighbors[n1*4+k1+1],Lattice[i].neighbors[n1*4+k1+2],Lattice[i].neighbors[n1*4+k1+3]);
		 // fprintf(out3," %e",Lattice[i].neighbors[n*4+k+0]);
	}

      sum /= size;
      fprintf(out2," %e %e %e\n",sum,sqrt(sum2/size - sum*sum),fabs(max_val - min_val));
      fprintf(out,"\n");
      fprintf(out3,"\n");
    }
  }

  fclose(out);
  fclose(out2);
  fclose(out3);
}


void WriteLSP_mpi()
{
  double *gis;
  // number of nodes in the input layer gives
  // the size of the Gis vector.
  // 2 for atom id and configuration id.
  int size = 2 + layer[0].nnodes;

  gis = new double [size];

  if (me == 0) {
    printf("Collecting Gis from cpus and writing to LSParam.dat .... [%s:%d]\n",
	   __FILE__, __LINE__);
    FILE *out, *out2;

    out = fopen("LSParam.dat", "w");

    if (!out) {
      printf("Error: cannot open LSParam.dat .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_NO_SUCH_FILE);
      exit(EXIT_FAILURE);
    }

    out2 = fopen("Gis_dist.dat","w");

    if (!out) {
      printf("Error: cannot open Gis_dist.dat .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_NO_SUCH_FILE);
      exit(EXIT_FAILURE);
    }


    // Get Gis from all processes.
    MPI_Status status;
    for (int i=0; i<Ntotal_bases - NAtoms; i++) { // Root must receive Ntotal_bases less NAtoms it holds.
      MPI_Recv(gis, size, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, world, &status);
      //int ndata;
      //MPI_Get_count(&status, MPI_DOUBLE, &ndata);
      //printf("master received %d doubles from slave %d ....\n", ndata, status.MPI_SOURCE);
      int idx = (int) gis[0];
      __GiList[idx].atomid = idx;
      __GiList[idx].gid = (int) gis[1];
      for (int j=0; j<size - 2; j++) {
	__GiList[idx].gilist[j] = gis[2 + j];
	//printf(" %f", __GiList[idx].gilist[j]);
      }
      //printf("\n");
    }
    //printf("master received all atoms from slaves ....\n");

    // Copy its Gis to GiList.
    for (int i=0; i<NAtoms; i++) {
      __GiList[atoms[i].atomid].atomid = atoms[i].atomid;
      __GiList[atoms[i].atomid].gid = atoms[i].gid;
      for (int j=0; j<size - 2; j++) {
	__GiList[atoms[i].atomid].gilist[j] = atoms[i].Gi_list[j];
      }
    }

 //---------LSPARAM LIST HEADER----------
  // Gi version, reference Gi and logistic function type.
  fprintf(out,"# %d %f %d - Gi version, reference Gi and type of logistic function.\n",GiMethod, REF_GI, LogisticFuncType);
  // number of chemical species in the system.
  fprintf(out,"# %d - number of chemical species in the system.\n", nChemSort);
  // element name and mass.
  for (int i=0; i<nChemSort; i++) {
    fprintf(out,"# %s %f\n",element[i],xmass[i]);
  }
  // flag and max. range
  fprintf(out,"# %d %f %f %f %f\n",NNetInit,MAX_RANGE,Rc,Hc,SS);
  // Write number of Legendre polynomials and orders.
  fprintf(out, "# %d", nLPOrders);
  for (int i=0; i<nLPOrders; i++) fprintf(out, " %d", LegendrePolyOrders[i]);
  fprintf(out, "\n");
  // Write number of Sigmas and values.
  fprintf(out,"# %d",nSigmas);
  for (int i=0; i<nSigmas; ++i) fprintf(out," %.2f",Sigmas[i]);
  fprintf(out,"\n");
  // Write flag and estimates of BO parameters.
  fprintf(out, "# %d", use_filter);
  for (int i=0; i<MAX_HB_PARAM; i++) fprintf(out, " %f", __BOP_EST[i]);
  fprintf(out, "\n");
  // Write information of NN architecture.
  fprintf(out,"# %d",nLayers);
  for (int l=0; l<nLayers; l++) {
    fprintf(out," %d",layer[l].nnodes);
  }
  fprintf(out,"\n");
  fprintf(out,"# %d -PotentialType \n",PotentialType);
  fprintf(out,"# %d -number of structures \n",nStruc);
  fprintf(out,"# %d -number of atoms \n",Ntotal_bases);
  fprintf(out,"#ATOM-ID GROUP-NAME GROUP_ID STRUCTURE_ID STRUCTURE_Natom STRUCTURE_E_DFT STRUCTURE_Vol \n");
	int k1;
	int j1;
	j1=0;
	k1=0;
    for (int i=0; i<Ntotal_bases; i++) {
      fprintf(out, "ATOM-%d %s %d %d %d %e %e\n", __GiList[i].atomid, Lattice[__GiList[i].gid].name, Lattice[__GiList[i].gid].cid, __GiList[i].gid, Lattice[__GiList[i].gid].nbas, Lattice[__GiList[i].gid].E0,Lattice[__GiList[i].gid].omega0);



      //fprintf(out," %s %d %d %d %e",Lattice[i].name,Lattice[i].cid,i,Lattice[i].nbas,Lattice[i].E0);

      fprintf(out, "%s ", "Gi");

      fprintf(out2, " %d %d %s", __GiList[i].atomid, __GiList[i].gid,
	      Lattice[__GiList[i].gid].name);
      double max_val, min_val;
      double sum = 0.0;
      double sum2 = 0.0;
      for (int k=0; k<size - 2; ++k) {
	double tmp = __GiList[i].gilist[k];
	fprintf(out, " %e", tmp);
	if (!k) max_val = min_val = tmp;
	else {
	  max_val = MAX(max_val,tmp);
	  min_val = MIN(min_val,tmp);
	}
	sum += tmp;
	sum2 += pow(tmp, 2);
      }
      sum /= (size - 2);
      fprintf(out2," %e %e %e %e\n", sum, sqrt(sum2/(size - 2) - sum*sum), max_val, min_val);
      fprintf(out,"\n");


    }
    fclose(out);
    fclose(out2);
  } else {
    // Pack data.
    for (int i=0; i<NAtoms; i++) {
      gis[0] = (double) atoms[i].atomid;
      gis[1] = (double) atoms[i].gid;
      for (int j=0; j<size - 2; j++) {
	gis[2 + j] = atoms[i].Gi_list[j];
      }
      MPI_Send(gis, size, MPI_DOUBLE, 0, me, world);
      //printf("slave %d sent %d doubles to master ....\n", me, size);
    }
    //printf("slave %d finished sending data ....\n", me);
  }

  if (gis) delete [] gis;
}
