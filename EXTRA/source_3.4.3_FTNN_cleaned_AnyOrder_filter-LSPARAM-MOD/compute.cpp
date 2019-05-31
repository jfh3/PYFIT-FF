#include <math.h> // cmath
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <assert.h>

#include "globals.h"
#include "mem.h"
#include "analytical.h"
#include "nr.h"
#include "NNetInterface.h"
#include "mpi_stuff.h"
//#include "amoeba.h"
//#include "amebsa.h"
#include "compute.h"
#include "write.h"
#include "crystal_struc.h"
#include "ran.h"

double Energy(const int *nn, const double *nlist, const int nbas, const VecDoub bop)
{
  double *Sij, Sijk, *zij, bij;
  double E, Ei, Ep, cut, cs;
  double rij, rik, rjk;
  double r, dx, dy, dz;
  int block_shft;
    
  E = 0.0;
  block_shft = 0;

  for (int i=0; i<nbas; i++) { /* loop over sites i */
    if (i) block_shft += 4*nn[i-1];
    Ei = 0.0;
    Ep = 0.0;
    Sij = new double [nn[i]];
    zij = new double [nn[i]];

    // initialize Sij and zij
    for (int m=0; m<nn[i]; m++) {
      Sij[m] = 1.0;
      zij[m] = 0.0;
    }

    // compute Sij
    for (int j=0; j<nn[i]; j++) { /* loop over sites j */
      rij = nlist[j*4 + block_shft];
      if (rij > Rc) continue;
      for (int k=0; k<nn[i]; k++) { /* loop over sites k */
	if (j == k) continue;
	rik = nlist[k*4 + block_shft];
	dx = nlist[j*4 + 1 + block_shft] - nlist[k*4 + 1 + block_shft];
	dy = nlist[j*4 + 2 + block_shft] - nlist[k*4 + 2 + block_shft];
	dz = nlist[j*4 + 3 + block_shft] - nlist[k*4 + 3 + block_shft];
	rjk = sqrt(dx*dx + dy*dy + dz*dz);
	r = rik + rjk - rij;
	if (r <= Rc) {
	  if (PotentialType == 2) cut = CutoffFunc(r,Rc,Hc);
	  else cut = CutoffFunc(r,Rc,bop[hc]);
	  Sijk = 1.0 - cut*exp(-pow(bop[lambda],2)*fabs(r));
	  Sij[j] *= Sijk;
	}
      }
      if (Sij[j] < 0.0) std::cout << " Found negative screening factor !!!\n";
    }

    // compute zij
    for (int j=0; j<nn[i]; j++) { /* loop over sites j */
      rij = nlist[j*4 + block_shft];
      if (rij > Rc) continue;
      for (int k=0; k<nn[i]; k++) { /* loop over sites k */
	if (j == k) continue;
	rik = nlist[k*4 + block_shft];
	if (rik > Rc) continue;
	dx = nlist[j*4 + 1 + block_shft] * nlist[k*4 + 1 + block_shft];
	dy = nlist[j*4 + 2 + block_shft] * nlist[k*4 + 2 + block_shft];
	dz = nlist[j*4 + 3 + block_shft] * nlist[k*4 + 3 + block_shft];
	cs = (dx + dy + dz)/rij/rik; /* cosine of angle between rij and rik */
	if (PotentialType == 2) cut = CutoffFunc(rik,Rc,Hc);
	else cut = CutoffFunc(rik,Rc,bop[hc]);
	zij[j] += Sij[k]*pow(bop[small_a],2)*pow(cs - bop[small_h],2)*cut;
      }
    }

    // compute energy of site i
    for (int j=0; j<nn[i]; j++) { /* loop over sites j */
      rij = nlist[j*4 + block_shft];
      if (rij > Rc) continue;
      bij = 1.0/sqrt(1.0 + zij[j]);
      if (PotentialType == 2) cut = CutoffFunc(rij,Rc,Hc);
      else cut = CutoffFunc(rij,Rc,bop[hc]);
      double A = bop[big_a];// * bop[big_a];
      double Alpha = bop[alpha];// * bop[alpha];
      double B = bop[big_b];// * bop[big_b];
      double Beta = bop[beta];// * bop[beta];
      Ei += (exp(A - Alpha*rij) - Sij[j]*bij*exp(B - Beta*rij))*cut;
      Ep += bij*Sij[j]*cut;
    }
    E += 0.5*Ei - bop[sigma]*sqrt(Ep);

    delete [] Sij;
    delete [] zij;
  }

  return E;
}


double iEnergy(const int nn, const double *nlist, const VecDoub &bop)
{
  double *Sij, **Sijk, *zij, *bij;
  double E, Ei, Ep, cut, **cs;
  double rij, rik, rjk;
  double r, dx, dy, dz;

  E = 0.0;
  Ei = 0.0;
  Ep = 0.0;
  Sij = new double [nn];
  zij = new double [nn];
  bij = new double [nn];
  Sijk = create(Sijk, nn, nn, "iEnergy:create()");
  cs = create(cs, nn, nn, "iEnergy:create()");

  // initialize Sij and zij
  for (int m=0; m<nn; m++) {
    Sij[m] = 1.0;
    zij[m] = 0.0;
    bij[m] = 0.0;
    for (int k=0; k<nn; k++) {
      Sijk[m][k] = 1.0;
      cs[m][k] = 1.0;
    }
  }

  // compute Sij
  for (int j=0; j<nn; j++) { /* loop over sites j */
    rij = nlist[j*4];
    if (rij > Rc) continue;
    for (int k=0; k<nn; k++) { /* loop over sites k */
      if (j == k) continue;
      rik = nlist[k*4];
      dx = nlist[j*4 + 1] - nlist[k*4 + 1];
      dy = nlist[j*4 + 2] - nlist[k*4 + 2];
      dz = nlist[j*4 + 3] - nlist[k*4 + 3];
      rjk = sqrt(dx*dx + dy*dy + dz*dz);
      r = rik + rjk - rij;
      if (r < Rc) {
	if (PotentialType == 2) cut = CutoffFunc(r, Rc, Hc);
	else cut = CutoffFunc(r, Rc, bop[hc]);
	Sijk[j][k] = 1.0 - cut * exp(-pow(bop[lambda], 2) * fabs(r));
	Sij[j] *= Sijk[j][k];
      }
    }
    if (Sij[j] < 0.0) std::cout << " Found negative screening factor !!!\n";
  }

  double dEi_dai_1{0.0};
  double dEi_dai_2{0.0};
  double dEi_dhi_1{0.0};
  double dEi_dhi_2{0.0};

  // compute zij
  for (int j=0; j<nn; j++) { /* loop over sites j */
    rij = nlist[j*4];
    double z_tmp{0.0};
    if (rij > Rc) continue;
    for (int k=0; k<nn; k++) { /* loop over sites k */
      if (j == k) continue;
      rik = nlist[k*4];
      if (rik > Rc) continue;
      dx = nlist[j*4 + 1] * nlist[k*4 + 1];
      dy = nlist[j*4 + 2] * nlist[k*4 + 2];
      dz = nlist[j*4 + 3] * nlist[k*4 + 3];
      cs[j][k] = (dx + dy + dz)/rij/rik; /* cosine of angle between rij and rik */
      if (PotentialType == 2) cut = CutoffFunc(rik, Rc, Hc);
      else cut = CutoffFunc(rik, Rc, bop[hc]);
      zij[j] += Sij[k] * pow(bop[small_a],2) * pow(cs[j][k] - bop[small_h],2) * cut;
      if (isLocalGrad) z_tmp += Sij[k] * (cs[j][k] - bop[small_h]) * cut;
    }
    if (isLocalGrad) {
      cut = CutoffFunc(rij, Rc, Hc);
      dEi_dai_1 += Sij[j] * exp(bop[big_b] - bop[beta]*rij) * cut * pow(1.0+zij[j],-1.5) * zij[j]/bop[small_a];
      dEi_dai_2 += Sij[j] * cut * pow(1.0+zij[j],-1.5) * zij[j]/bop[small_a];
      dEi_dhi_1 += Sij[j] * exp(bop[big_b] - bop[beta]*rij) * cut * pow(1.0+zij[j],-1.5) * z_tmp;
      dEi_dhi_2 += Sij[j] * cut * pow(1.0+zij[j],-1.5) * z_tmp;
    }
  }

  // compute energy of site i
  for (int j=0; j<nn; j++) { /* loop over sites j */
    rij = nlist[j*4];
    if (rij > Rc) continue;
    bij[j] = 1.0/sqrt(1.0 + zij[j]);
    if (PotentialType == 2) cut = CutoffFunc(rij,Rc,Hc);
    else cut = CutoffFunc(rij,Rc,bop[hc]);
    double A = bop[big_a];// * bop[big_a];
    double Alpha = bop[alpha];// * bop[alpha];
    double B = bop[big_b];// * bop[big_b];
    double Beta = bop[beta];// * bop[beta];
    double repul = exp(A - Alpha*rij)*cut;
    double attr = Sij[j]*bij[j]*exp(B - Beta*rij)*cut;
    Ei += repul - attr;
    //Ei += (exp(A - Alpha*rij) - Sij[j]*bij*exp(B - Beta*rij))*cut;
    if (isLocalGrad) {
      MatA[0][big_a] += 0.5*repul; // derivative w.r.t. Ai
      MatA[0][alpha] += -0.5*rij*repul; // derivative w.r.t. alpha
      MatA[0][big_b] += -0.5*attr; // derivative w.r.t. Bi
      MatA[0][beta] += 0.5*rij*attr; // derivative w.r.t. beta
    }
    Ep += bij[j]*Sij[j]*cut;
  }

  E += 0.5*Ei - bop[sigma]*sqrt(Ep);

  if (isLocalGrad) {
    MatA[0][sigma] = -sqrt(Ep); // derivative w.r.t. sigma
    MatA[0][small_a] += 0.5 * dEi_dai_1 + 0.5 * bop[sigma]/sqrt(Ep) * dEi_dai_2; // partial derivative w.r.t. ai
    MatA[0][small_h] += -0.5 * pow(bop[small_a],2) * (dEi_dhi_1 + bop[sigma]/sqrt(Ep) * dEi_dhi_2); // partial derivative w.r.t. hi

    double dEi_dli_1{0.0};
    double dEi_dli_2{0.0};

    // derivative w.r.t. lambda
    for (int j=0; j<nn; j++) { /* loop over sites j */
      rij = nlist[j*4];
      double dSij_dli{0.0};
      if (rij > Rc) continue;
      double tmp{0.0};
      for (int k=0; k<nn; k++) { /* loop over sites k */
	if (j == k) continue;
	rik = nlist[k*4];
	dx = nlist[j*4 + 1] - nlist[k*4 + 1];
	dy = nlist[j*4 + 2] - nlist[k*4 + 2];
	dz = nlist[j*4 + 3] - nlist[k*4 + 3];
	rjk = sqrt(dx*dx + dy*dy + dz*dz);
	r = rik + rjk - rij;
	if (r < Rc) {
	  if (PotentialType == 2) cut = CutoffFunc(r,Rc,Hc);
	  else cut = CutoffFunc(r,Rc,bop[hc]);
	  dSij_dli += 2.0 * bop[lambda] * Sij[j]/Sijk[j][k] * r *
	      exp(-pow(bop[lambda],2)*r) * cut; // sum over k.
	}
	if (rik < Rc) {
	  double dSik_dli{0.0};
	  for (int l=0; l<nn; l++) { // loop over l.
	    if (l == j) continue;
	    double ril = nlist[l*4];
	    dx = nlist[k*4 + 1] - nlist[l*4 + 1];
	    dy = nlist[k*4 + 2] - nlist[l*4 + 2];
	    dz = nlist[k*4 + 3] - nlist[l*4 + 3];
	    double rkl = sqrt(dx*dx + dy*dy + dz*dz);
	    r = ril + rkl - rik;
	    if (r < Rc) {
	      if (PotentialType == 2) cut = CutoffFunc(r,Rc,Hc);
	      else cut = CutoffFunc(r,Rc,bop[hc]);
	      dSik_dli += 2.0 * bop[lambda] * Sij[k]/Sijk[k][l] * r *
		  exp(-pow(bop[lambda],2)*r) *cut;
	    }
	  }
	  if (PotentialType == 2) cut = CutoffFunc(rik,Rc,Hc);
	  else cut = CutoffFunc(rik,Rc,bop[hc]);
	  tmp += pow(cs[j][k]-bop[small_h],2) * cut * dSik_dli; // sum over k.
	}
      }
      if (PotentialType == 2) cut = CutoffFunc(rij,Rc,Hc);
      else cut = CutoffFunc(rij,Rc,bop[hc]);
      dEi_dli_1 += -0.5 * bij[j] * exp(bop[big_b] - bop[beta] * rij) * cut * dSij_dli; // from attractive part of Ei
      dEi_dli_1 += -0.5 * bop[sigma] / sqrt(Ep) * bij[j] * cut * dSij_dli; // from E_i^p part of Ei
      dEi_dli_2 += 0.25 * pow(bop[small_a],2) * Sij[j] * exp(bop[big_b]-bop[beta]*rij) *
	  pow(1.0+zij[j],-1.5) * cut * tmp;
      dEi_dli_2 += 0.25 * bop[sigma] * pow(bop[small_a],2) / sqrt(Ep) * Sij[j] *
	  pow(1.0+zij[j],-1.5) * cut * tmp;
    }
    MatA[0][lambda] += dEi_dli_1 + dEi_dli_2;
  }

  delete [] Sij;
  delete [] zij;
  delete [] bij;
  destroy(Sijk);
  destroy(cs);

  return E;
}

double Atomic_Eng(const int *nn, const double *nlist,
		  const int Basis_Id, const VecDoub bop)
{
  double *Sij, Sijk, *zij, bij;
  double E, Ei, Ep, cut, cs;
  double rij, rik, rjk;
  double r, dx, dy, dz;
  int block_shft;

  block_shft = 0;

  // compute cummulative data shift
  for (int i=0; i<Basis_Id; i++) block_shft += 4*nn[i];

  Ei = 0.0;
  Ep = 0.0;
  Sij = new double [nn[Basis_Id]];
  zij = new double [nn[Basis_Id]];

  // initialize Sij and zij
  for (int m=0; m<nn[Basis_Id]; m++) {
    Sij[m] = 1.0;
    zij[m] = 0.0;
  }

  // compute Sij
  for (int j=0; j<nn[Basis_Id]; j++) { /* loop over sites j */
    rij = nlist[j*4 + block_shft];
    if (rij > Rc) continue;
    for (int k=0; k<nn[Basis_Id]; k++) { /* loop over sites k */
      if (j == k) continue;
      rik = nlist[k*4 + block_shft];
      dx = nlist[j*4 + 1 + block_shft] - nlist[k*4 + 1 + block_shft];
      dy = nlist[j*4 + 2 + block_shft] - nlist[k*4 + 2 + block_shft];
      dz = nlist[j*4 + 3 + block_shft] - nlist[k*4 + 3 + block_shft];
      rjk = sqrt(dx*dx + dy*dy + dz*dz);
      r = rik + rjk - rij;
      if (r <= Rc) {
	if (PotentialType == 2) cut = CutoffFunc(r,Rc,Hc);
	else cut = CutoffFunc(r,Rc,bop[hc]);
	Sijk = 1.0 - cut*exp(-pow(bop[lambda],2)*fabs(r));
	Sij[j] *= Sijk;
      }
    }
    if (Sij[j] < 0.0) std::cout << " Found negative screening factor !!!\n";
    //std::cout << j+1 << " " << Sij[j] << "\n";
  }

  // compute zij
  for (int j=0; j<nn[Basis_Id]; j++) { /* loop over sites j */
    rij = nlist[j*4 + block_shft];
    if (rij > Rc) continue;
    for (int k=0; k<nn[Basis_Id]; k++) { /* loop over sites k */
      if (j == k) continue;
      rik = nlist[k*4 + block_shft];
      if (rik > Rc) continue;
      dx = nlist[j*4 + 1 + block_shft] * nlist[k*4 + 1 + block_shft];
      dy = nlist[j*4 + 2 + block_shft] * nlist[k*4 + 2 + block_shft];
      dz = nlist[j*4 + 3 + block_shft] * nlist[k*4 + 3 + block_shft];
      cs = (dx + dy + dz)/rij/rik; /* cosine of angle between rij and rik */
      if (PotentialType == 2) cut = CutoffFunc(rik,Rc,Hc);
      else cut = CutoffFunc(rik,Rc,bop[hc]);
      zij[j] += Sij[k]*pow(bop[small_a],2)*pow(cs - bop[small_h],2)*cut;
    }
  }

  // compute energy of site i
  for (int j=0; j<nn[Basis_Id]; j++) { /* loop over sites j */
    rij = nlist[j*4 + block_shft];
    if (rij > Rc) continue;
    bij = 1.0/sqrt(1.0 + zij[j]);
    if (PotentialType == 2) cut = CutoffFunc(rij,Rc,Hc);
    else cut = CutoffFunc(rij,Rc,bop[hc]);
    double A = bop[big_a];// * bop[big_a];
    double Alpha = bop[alpha];// * bop[alpha];
    double B = bop[big_b];// * bop[big_b];
    double Beta = bop[beta];// * bop[beta];
    Ei += (exp(A - Alpha*rij) - Sij[j]*bij*exp(B - Beta*rij))*cut;
    Ep += bij*Sij[j]*cut;
  }
  E = 0.5*Ei - bop[sigma]*sqrt(Ep);

  delete [] Sij;
  delete [] zij;

  return E;
}

void DFP_minimize()
{
  if (me == 0) {
    int itr;
    int total_itr;
    double f, test_dev;
    double stime, etime;
    double itr_time;
    FILE *fp;
    char buf[256];

    isLocalGrad = 0; // This will be updated by df(,,);
    itr = total_itr = 0;
    if (NNetInit) NNetInit = 0;

    //printf(" Number of configurations: %d\n",nStruc);
    printf(" Stage(s) Default_iter Actual_iter Dev Time(s) Test_Error\n");
    printf(" --------------------------------------------------------\n");
    stime = MPI_Wtime();
    for (int i=0; i<Nstg; i++) {
      itr = iter0;
      itr_time = MPI_Wtime();
      //double begin = MPI_Wtime();
      //test_dev = compute_error(TestSet,nTestSize,ParamVec);
      //printf(" time to compute error in Test set is %f s.\n",MPI_Wtime()-begin);
      //exit(0);
      dfpmin(ParamVec, gTol, itr, f, Funk);
      if (PotentialType == 2) WriteHBparam();
      f -= BOPconstraint + NNconstraint + HBconstraint + HBconstraint2;
      double tmp_x;
      // Returns number of elements greater than
      // the second argument in the function; also
      // returns the value of an element with max. absolute
      // value greater than the second argument;
      int tmp_n = compareAbsMaxVec(ParamVec,5.0,tmp_x);
      if (strcmp(testfile,"none") != 0) {
        test_dev = compute_error(TestSet,nTestSize,ParamVec);
        printf(" %5d %5d %5d %e %f %e %f %f %f %f (%d,%f)\n",i+1,iter0,itr+1,sqrt(f),
               MPI_Wtime()-itr_time,test_dev,sqrt(BOPconstraint),
               sqrt(NNconstraint),sqrt(HBconstraint),sqrt(HBconstraint2),tmp_n,tmp_x);
      }
      else printf(" %5d %5d %5d %e %f NA %f %f %f %f (%d,%f)\n",i+1,iter0,itr+1,sqrt(f),
                  MPI_Wtime() - itr_time,sqrt(BOPconstraint),sqrt(NNconstraint),
                  sqrt(HBconstraint),sqrt(HBconstraint2),tmp_n,tmp_x);
      total_itr += itr + 1;

      if (PotentialType == 0) {
        sprintf(buf,"param.%d.dat",i+1);
        fp = fopen(buf,"w");
        WriteBOPParam(fp);
        fclose(fp);
      }

      if (PotentialType == 1 || PotentialType == 2) {
        sprintf(buf,"param.%d.dat",i+1);
        fp = fopen(buf,"w");
        // write weights and biases of hidden layers and the output layer.
        WriteNNetParam(fp);
        fclose(fp);

        // write weights and biases of only hidden layers.
        sprintf(buf,"HLparam.%d.dat",i+1);
        fp = fopen(buf,"w");
        WriteHiddenLayerParam(fp);
        fclose(fp);

        if (noGi != -1) {
          // write outputs from the last hidden layer before the output layer.
          sprintf(buf,"mgi.%d.out",i+1);
          fp = fopen(buf,"w");
          WriteLastLayerOutputs(fp);
          fclose(fp);
        }
      }
    }
    etime = MPI_Wtime();
    printf(" --------------------------------------------------------\n");
    printf(" It took %f seconds to complete %d iterations.\n",
           etime - stime,total_itr);

    f = Funk(ParamVec);
    f -= BOPconstraint + NNconstraint + HBconstraint + HBconstraint2;
    printf(" total deviation = %e\n",sqrt(f));
    if (strcmp(testfile,"none") != 0)
      printf(" test_error = %e\n",compute_error(TestSet,nTestSize,ParamVec));
    printf(" BOP constraint = %f\n",sqrt(BOPconstraint));
    printf(" NN constraint = %f\n",sqrt(NNconstraint));
    printf(" HB constraint = %f\n",sqrt(HBconstraint));
    printf(" HB2 constraint = %f\n",sqrt(HBconstraint2));
    double tmp_x;
    int tmp_n = compareAbsMaxVec(ParamVec,5.0,tmp_x); // returns number of elements greater than
    // the second argument in the function; also
    // returns the value of an element with max. absolute
    // value greater than the second argument;
    printf(" %d parameters are > 5.0 (max. = %f)\n",tmp_n,tmp_x);
    if (!Nstg) {
      if (PotentialType == 1 || PotentialType == 2) {
        // write weights and biases of only hidden layers.
        fp = fopen("HLparam.dat","w");
        WriteHiddenLayerParam(fp);
        fclose(fp);
        // write outputs from the last hidden layer before the output layer.
        fp = fopen("mgi.out","w");
        WriteLastLayerOutputs(fp);
        fclose(fp);
      }
    }

    // send dummy data to all slave nodes with tag 0 here.
    mpi_send_dummy();
  } else {
    mpi_exchange_slaves_nnet ();
  }
}

double Funk(VecDoub &pin)
{
  if (me) {
    printf("Error: only master should run this routine: in file %s at line %d\n",
	   __FILE__,__LINE__);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  double dev = 0.0;
  BOPconstraint = 0.0;
  NNconstraint = 0.0;
  HBconstraint = 0.0;
  HBconstraint2 = 0.0;

  //if (nprocs > 1) {
    mpi_exchange_master_nnet(pin);
    for (int i=0; i<nStruc; i++) {
      //printf("configuration %d: %f\n", i, Lattice[i].E);
      dev += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas, 2) * Lattice[i].w;
    }
    if (PotentialType == 2) HBconstraint2 = compute_hb_constraint2();
  /*} else {
    // compute energies of structures and deviation.
    switch (PotentialType) {
    case 0: // Only for BOP
      for (int i=0; i<nStruc; i++) {
	Lattice[i].E = Energy(Lattice[i].nneighbors,
			      Lattice[i].neighbors,
			      Lattice[i].nbas,
			      pin);
	dev += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas,2) * Lattice[i].w;
      }
      break;
    case 1: // Only for NN
      // If using neural network, distribute parameters to
      // respective weights and biases.
      PartitionNNetParams(pin);
      for (int i=0; i<nStruc; i++) {
	Lattice[i].E = NNET_Eng(Lattice[i].G_local,
				Lattice[i].nbas);
	dev += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas,2) * Lattice[i].w;
      }
      break;
    case 2: // For PINN
      PartitionNNetParams(pin);
      for (int i=0; i<nStruc; i++) {
	Lattice[i].E = NNET_Eng(Lattice[i].G_local,
				i,
				Lattice[i].nbas);
	dev += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas,2) * Lattice[i].w;
      }
      break;
    default:
      printf("Error: potential type not currently supported: in file %s at line %d\n",
	     __FILE__,__LINE__);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    if (PotentialType == 2) HBconstraint2 = compute_hb_constraint2(pin);
  }*/

  if (PotentialType == 0) {
    BOPconstraint += specific_bop_constraint(pin);
    BOPconstraint += mean_sqr(pin, CONST_BOP);
  }

  if (PotentialType == 1 || PotentialType == 2)
    NNconstraint = mean_sqr(pin, CONST_NN); // constraint for weights and biases.

  if (PotentialType == 2) HBconstraint /= Ntotal_bases;

  return dev/nStruc + BOPconstraint + NNconstraint
      + HBconstraint + HBconstraint2;
}

/*double Funk_array(const double *pin, const int n)
{
  if (me) {
    errmsg("only master should run this routine",FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  double dev = 0.0;
  BOPconstraint = 0.0;
  NNconstraint = 0.0;
  HBconstraint = 0.0;
  HBconstraint2 = 0.0;

  VecDoub pvec(n,pin);

  if (nprocs > 1) {
    mpi_exchange_master_nnet(pvec);
    for (int i=0; i<nStruc; i++) {
      dev += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas,2) * Lattice[i].w;
    }
    //double start = MPI_Wtime();
    if (PotentialType == 2) HBconstraint2 = compute_hb_constraint2();
    //printf(" time to compute HBconstraint2(%.16e) is %f s.\n",sqrt(HBconstraint2),MPI_Wtime()-start);
  } else {
    // compute energies of structures and deviation.
//#pragma omp parallel for schedule(dynamic) reduction(+:dev)
    for (int i=0; i<nStruc; i++) {
      switch (PotentialType) {
      case 0: // Only for BOP
	Lattice[i].E = Energy(Lattice[i].nneighbors,Lattice[i].neighbors,
			      Lattice[i].nbas,pvec);
	break;
      case 1: // Only for NNET
	// If using neural network, distribute parameters to
	// respective weights and biases.
	PartitionNNetParams(pvec);
	Lattice[i].E = NNET_Eng(Lattice[i].G_local,Lattice[i].nbas);
	break;
      case 2: // For NNET + BOP
	PartitionNNetParams(pvec);
	Lattice[i].E = NNET_Eng(Lattice[i].G_local,i,Lattice[i].nbas);
	break;
      default:
	errmsg("potential type is not currently supported",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      dev += pow((Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas,2) * Lattice[i].w;
    }
    //double start = MPI_Wtime();
    if (PotentialType == 2) HBconstraint2 = compute_hb_constraint2(pvec);
    //printf(" time to compute HBconstraint2 = %f s.\n",MPI_Wtime()-start);
  }

  if (PotentialType == 0) {
    BOPconstraint += specific_bop_constraint(pvec);
    BOPconstraint += mean_sqr(pvec,CONST_BOP);
  }

  if (PotentialType == 1 || PotentialType == 2)
    NNconstraint = mean_sqr(pvec,CONST_NN); // constraint for weights and biases.

  if (PotentialType == 2) HBconstraint /= Ntotal_bases;

  return dev/nStruc + BOPconstraint + NNconstraint
      + HBconstraint + HBconstraint2;
}*/

void ComputeLocalStrucParam(Struc_Data *&data, const int nSet)
{
  int block_shft;
  double rij, rik;
  int ni;
  double Pl[7];
  double tmp;
  double xj, yj, zj;
  double xk, yk, zk;
  double gmin = REF_GI;
  double rc;

  rc = 1.5 * Rc;
  if (PotentialType == 1) rc = Rc;
  Pl[0] = 1.0;

  for (int l=0; l<nSet; l++) { // loop over configurations
    block_shft = 0;
    ni = data[l].nbas*nSigmas*MAX_LSP;
    data[l].G_local = create(data[l].G_local,ni,"ComputeLocalStrucParam()");
    // initialize G_local
    for (int k=0; k<ni; k++) data[l].G_local[k] = 0.0;
    for (int i=0; i<data[l].nbas; i++) { // loop over sites i
      if (i) block_shft += 4*data[l].nneighbors[i-1];
      for (int n=0; n<nSigmas; n++) { // loop over Sigmas
	//for (int m=0; m<5; m++) { // loop over Gis
	for (int j=0; j<data[l].nneighbors[i]; j++) { // loop over sites j
	  rij = data[l].neighbors[j*4 + block_shft];
	  if (rij > rc) continue;
	  for (int k=0; k<data[l].nneighbors[i]; k++) { // loop over sites k
	    rik = data[l].neighbors[k*4 + block_shft];
	    if (rik > rc) continue;
	    xj = data[l].neighbors[j*4 + 1 + block_shft];
	    yj = data[l].neighbors[j*4 + 2 + block_shft];
	    zj = data[l].neighbors[j*4 + 3 + block_shft];
	    xk = data[l].neighbors[k*4 + 1 + block_shft];
	    yk = data[l].neighbors[k*4 + 2 + block_shft];
	    zk = data[l].neighbors[k*4 + 3 + block_shft];
	    double costheta = (xj*xk + yj*yk + zj*zk)/(rij*rik);
	    tmp = exp(-pow((rij-Sigmas[n])/SS,2))*CutoffFunc(rij, rc, Hc);
	    tmp *= exp(-pow((rik-Sigmas[n])/SS,2))*CutoffFunc(rik, rc, Hc)/16.0; //pow(Sigmas[n],2);
	    Pl[1] = costheta;
	    for (int m=1; m<6; m++) Pl[m+1] = ((2.0*m+1.0)*costheta*Pl[m] - m*Pl[m-1])/(m+1);
	    data[l].G_local[i*nSigmas*MAX_LSP + n] += tmp; // Gi(0); m = 0;
	    data[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n] += Pl[1]*tmp; // Gi(1); m = 1;
	    data[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n] += Pl[2]*tmp; // Gi(2); m = 2;
	    data[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n] += Pl[4]*tmp; // Gi(3); m = 3;
	    data[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n] += Pl[6]*tmp; // Gi(4); m = 4;

	    // only for debugging purpose
	    /*printf("config: %d basis: %d Gi0(%d) = %f\n",
						l,i,n,data[l].G_local[i*nSigmas*MAX_LSP + n]);
						printf("config: %d basis: %d Gi1(%d) = %f\n",
						l,i,n,data[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n]);
						printf("config: %d basis: %d Gi2(%d) = %f\n",
						l,i,n,data[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n]);
						printf("config: %d basis: %d Gi3(%d) = %f\n",
			   l,i,n,data[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n]);
						printf("config: %d basis: %d Gi4(%d) = %f\n",
						l,i,n,data[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n]);*/
	  }
	}
	// recondition Gis
	if (gmin != 0.0) {
	  data[l].G_local[i*nSigmas*MAX_LSP + n] += gmin;
	  data[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n] += gmin;
	  data[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n] += gmin;
	  data[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n] += gmin;
	  data[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n] += gmin;
	}
	switch(GiMethod) {
	case 0:
	  break;
	case 1:
	  data[l].G_local[i*nSigmas*MAX_LSP + n] = log(data[l].G_local[i*nSigmas*MAX_LSP + n]);
	  data[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n] = log(data[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n]);
	  data[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n] = log(data[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n]);
	  data[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n] = log(data[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n]);
	  data[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n] = log(data[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n]);
	  break;
	default:
	  printf("Error: Gi method:%d not supported .. from file %s at line %d\n",
		 GiMethod,__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,errcode);
	  exit(EXIT_FAILURE);
	}
      }
    }
  }
}

/*double compute_hb_constraint2(VecDoub &pin)
{
  if (PotentialType == 2) {
    double **HBParam_map;
    HBParam_map = create(HBParam_map,Ntotal_bases,MAX_HB_PARAM,
			 "compute_hb_constraint2():create");
    PartitionNNetParams(pin);
    int size, ncount;
    size = nSigmas*MAX_LSP;
    ncount = 0;

    for (int l=0; l<nStruc; l++) { // loop over configurations.
      for (int i=0; i<Lattice[l].nbas; i++) { // loop over bases of each configuration.
	// Update input to NN.
	for (int j=0; j<size; j++) layer[0].vsum[j] = Lattice[l].G_local[i*size + j];
	// Compute local BOP parameters.
	evaluate_nnet();
	int k = nLayers - 1;
	if (layer[k].nnodes != MAX_HB_PARAM) {
	  printf("Error: check the condition here from file %s at line %d\n",__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,errcode);
	  exit(EXIT_FAILURE);
	}
	for (int m=0; m<layer[k].nnodes; m++) HBParam_map[ncount][m] = layer[k].vsum[m];
	ncount++;
      }
    }

    if (ncount != Ntotal_bases) {
      printf("Error: check the condition here from file %s at line %d\n",__FILE__,__LINE__);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    // Write hybrid BOP parameters of all atoms.
    if (Nstg == 0) {
      FILE *out;
      out = fopen("hb_BOP.dat","w");
      for (int i=0; i<Ntotal_bases; i++) {
	for (int j=0; j<MAX_HB_PARAM; j++) fprintf(out," %e",HBParam_map[i][j]);
	fprintf(out,"\n");
      }
      fclose(out);
    }


    double sum[MAX_HB_PARAM];
    for (int i=0; i<MAX_HB_PARAM; i++) sum[i] = 0.0;

    // Average each columns.
    for (int j=0; j<MAX_HB_PARAM; j++) {
      for (int i=0; i<Ntotal_bases; i++) {
	sum[j] += HBParam_map[i][j];
      }
      sum[j] /= Ntotal_bases;
    }

    // Compute max. deviation from mean for each column and sum them.
    double sum_errors = 0.0;
    for (int i=0; i<Ntotal_bases; i++) {
      for (int j=0; j<MAX_HB_PARAM; j++) {
	sum_errors += pow(HBParam_map[i][j]-sum[j],2);
      }
    }

    destroy(HBParam_map);

    return CONST_HB2*sum_errors/(Ntotal_bases*MAX_HB_PARAM);
  }

  return 0.0;
}*/

/*void icompute_hb_constraint2(VecDoub &pin)
{
  if (PotentialType == 2) {
    //double **HBParam_map;
    //HBParam_map = create(HBParam_map,NAtoms,MAX_HB_PARAM,"icompute_hb_constraint2():create");
    for (int i=0; i<max_num_atoms_per_proc*MAX_HB_PARAM; i++) local_max_pi[i] = 0.0;
    PartitionNNetParams(pin);
    int size;
    size = nSigmas*MAX_LSP;

    for (int i=0; i<NAtoms; i++) { // loop over bases of each configuration.
      // Update input to NN.
      for (int j=0; j<size; j++) layer[0].vsum[j] = atoms[i].Gi_list[j];
      // Compute local BOP parameters.
      evaluate_nnet();
      int k = nLayers - 1;
      if (layer[k].nnodes != MAX_HB_PARAM) {
	errmsg("check the condition here",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      for (int m=0; m<layer[k].nnodes; m++) local_max_pi[i*MAX_HB_PARAM + m] = layer[k].vsum[m];
    }

    // Average each columns.
    double sum[MAX_HB_PARAM];
    for (int i=0; i<MAX_HB_PARAM; i++) sum[i] = 0.0;
    for (int j=0; j<MAX_HB_PARAM; j++) {
      for (int i=0; i<NAtoms; i++) {
	sum[j] += HBParam_map[i][j];
      }
      sum[j] /= NAtoms;
    }

    // Compute max. deviation from mean for each column.
    for (int i=0; i<NAtoms; i++) {
      for (int j=0; j<MAX_HB_PARAM; j++) {
	local_max_pi[j] += pow(HBParam_map[i][j]-sum[j],2);
      }
    }

				destroy(HBParam_map);
    //CONST_HB2*sum_errors/MAX_HB_PARAM;
  }
}
*/

/*double icompute_hb_constraint2(const int strucId, VecDoub &pin)
{
  if (PotentialType == 2) {

    double **HBParam_map;

    HBParam_map = create(HBParam_map,Lattice[strucId].nbas,MAX_HB_PARAM,
			 "icompute_hb_constraint2(int,VecDoub):create");

    PartitionNNetParams(pin);

    int size, ncount;

    size = nSigmas*MAX_LSP;
    ncount = 0;

    for (int i=0; i<Lattice[strucId].nbas; i++) { // loop over bases of a configuration.
      // Update input to NN.
      for (int j=0; j<size; j++) layer[0].vsum[j] = Lattice[strucId].G_local[i*size + j];
      // Compute local BOP parameters.
      evaluate_nnet();
      int k = nLayers - 1;

      if (layer[k].nnodes != MAX_HB_PARAM) {
	errmsg("check the condition here",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      for (int m=0; m<layer[k].nnodes; m++) HBParam_map[ncount][m] = layer[k].vsum[m];
      ncount++;
    }

    if (ncount != Lattice[strucId].nbas) {
      errmsg("check the condition here",FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    // Average each columns.
    double sum[MAX_HB_PARAM];
    for (int i=0; i<MAX_HB_PARAM; i++) sum[i] = 0.0;
    for (int j=0; j<MAX_HB_PARAM; j++) {
      for (int i=0; i<Lattice[strucId].nbas; i++) {
	sum[j] += HBParam_map[i][j];
      }
      sum[j] /= Lattice[strucId].nbas;
    }

    // Compute max. deviation from mean for each column and sum them.
    double sum_errors = 0.0;
    for (int i=0; i<Lattice[strucId].nbas; i++) {
      for (int j=0; j<MAX_HB_PARAM; j++) {
	sum_errors += pow(HBParam_map[i][j]-sum[j],2);
      }
    }

    destroy(HBParam_map);

    return CONST_HB2*sum_errors/(Lattice[strucId].nbas*MAX_HB_PARAM);
  }

  return 0.0;
}*/

double compute_error(Struc_Data *&data, const int size, VecDoub &pin)
{
  double dev = 0.0;

  if (data) {
    // Compute energies of configurations.
    for (int i=0; i<size; i++) {
      double sum = 0.0, f0;
      switch (PotentialType) {
      case 0: // Only for BOP
	data[i].E = Energy(data[i].nneighbors,data[i].neighbors,
			   data[i].nbas,pin);
	break;
      case 1: // Only for NNET
	// If using neural network, distribute parameters to
	// respective weights and biases.
	PartitionNNetParams(pin);
	data[i].E = NNET_Eng(data[i].G_local,data[i].nbas);
	break;
      case 2: // For NNET + BOP
	PartitionNNetParams(pin);
	for (int k=0; k<data[i].nbas; k++) {
	  // update biases of layer[0]
	  for (int j=0; j<nSigmas*MAX_LSP; j++)
	    layer[0].vsum[j] = data[i].G_local[k*nSigmas*MAX_LSP + j];
	  NNetOutput(data[i].nneighbors,data[i].neighbors,k,f0,0);
	  sum += f0;
	}
	data[i].E = sum;
	break;
      default:
	errmsg("potential type is not currently supported",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      dev += pow((data[i].E - data[i].E0)/data[i].nbas,2) * data[i].w;
    }
  }

  return sqrt(dev/size);
}

void ComputeLocalStrucParam()
{
  int block_shft;
  double rij, rik;
  int ni;
  double Pl[7];
  double tmp;
  double xj, yj, zj;
  double xk, yk, zk;
  double gmin = REF_GI;
  double rc;

  rc = 1.5 * Rc;
  if (PotentialType == 1) rc = Rc;

  Pl[0] = 1.0;

  for (int l=0; l<nStruc; l++) { // loop over configurations
    block_shft = 0;
    ni = Lattice[l].nbas*nSigmas*MAX_LSP;
    Lattice[l].G_local = create(Lattice[l].G_local,ni,"ComputeLocalStrucParam()");
    // initialize G_local
    for (int k=0; k<ni; k++) Lattice[l].G_local[k] = 0.0;
    for (int i=0; i<Lattice[l].nbas; i++) { // loop over sites i
      if (i) block_shft += 4*Lattice[l].nneighbors[i-1];
      for (int n=0; n<nSigmas; n++) { // loop over Sigmas
	//for (int m=0; m<5; m++) { // loop over Gis
	for (int j=0; j<Lattice[l].nneighbors[i]; j++) { // loop over sites j
	  rij = Lattice[l].neighbors[j*4 + block_shft];
	  if (rij > rc) continue;
	  for (int k=0; k<Lattice[l].nneighbors[i]; k++) { // loop over sites k
	    rik = Lattice[l].neighbors[k*4 + block_shft];
	    if (rik > rc) continue;
	    xj = Lattice[l].neighbors[j*4 + 1 + block_shft];
	    yj = Lattice[l].neighbors[j*4 + 2 + block_shft];
	    zj = Lattice[l].neighbors[j*4 + 3 + block_shft];
	    xk = Lattice[l].neighbors[k*4 + 1 + block_shft];
	    yk = Lattice[l].neighbors[k*4 + 2 + block_shft];
	    zk = Lattice[l].neighbors[k*4 + 3 + block_shft];
	    double costheta = (xj*xk + yj*yk + zj*zk)/(rij*rik);
	    //if (fabs(costheta) > 1.0) std::cerr << costheta << "\n";
	    tmp = exp(-pow((rij-Sigmas[n])/SS,2))*CutoffFunc(rij, rc, Hc);
	    tmp *= exp(-pow((rik-Sigmas[n])/SS,2))*CutoffFunc(rik, rc, Hc)/16.0; //pow(Sigmas[n],2);
	    Pl[1] = costheta;
	    for (int m=1; m<6; m++) Pl[m+1] = ((2.0*m+1.0)*costheta*Pl[m] - m*Pl[m-1])/(m+1);
	    Lattice[l].G_local[i*nSigmas*MAX_LSP + n] += tmp; // Gi(0); m = 0;
	    Lattice[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n] += Pl[1]*tmp; // Gi(1); m = 1;
	    Lattice[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n] += Pl[2]*tmp; // Gi(2); m = 2;
	    Lattice[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n] += Pl[4]*tmp; // Gi(3); m = 3;
	    Lattice[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n] += Pl[6]*tmp; // Gi(4); m = 4;

	    // only for debugging purpose
	    /*printf("config: %d basis: %d Gi0(%d) = %f\n",
			   l,i,n,Lattice[l].G_local[i*nSigmas*MAX_LSP + n]);
		    printf("config: %d basis: %d Gi1(%d) = %f\n",
			   l,i,n,Lattice[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n]);
		    printf("config: %d basis: %d Gi2(%d) = %f\n",
			   l,i,n,Lattice[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n]);
		    printf("config: %d basis: %d Gi3(%d) = %f\n",
			   l,i,n,Lattice[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n]);
		    printf("config: %d basis: %d Gi4(%d) = %f\n",
			   l,i,n,Lattice[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n]);*/

	  }
	}
	// recondition Gis
	if (gmin != 0.0) {
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + n] += gmin;
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n] += gmin;
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n] += gmin;
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n] += gmin;
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n] += gmin;
	}
	switch(GiMethod) {
	case 0:
	  break;
	case 1:
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + n] = log(Lattice[l].G_local[i*nSigmas*MAX_LSP + n]);
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n] = log(Lattice[l].G_local[i*nSigmas*MAX_LSP + nSigmas + n]);
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n] = log(Lattice[l].G_local[i*nSigmas*MAX_LSP + 2*nSigmas + n]);
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n] = log(Lattice[l].G_local[i*nSigmas*MAX_LSP + 3*nSigmas + n]);
	  Lattice[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n] = log(Lattice[l].G_local[i*nSigmas*MAX_LSP + 4*nSigmas + n]);
	  break;
	default:
	  printf("Error: Gi method:%d not supported .. from file %s at line %d\n",
		 GiMethod,__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,errcode);
	  exit(EXIT_FAILURE);
	}
      }
    }
  }
}

void iCompute_LSP (const int *nn, const double *nlist,
		   const int nb, double *gi)
{
  int block_shft;
  double rij, rik;
  int ni;
  double *Pl;
  double xj, yj, zj;
  double xk, yk, zk;
  double tmp;
  double gmin = REF_GI;
  double rc;

  rc = Rc;

  if (GiMethod == 3 || GiMethod == 4) rc = 1.5 * Rc;

  //if (PotentialType == 0 || PotentialType == 1) rc = Rc;

  Pl = new double [LegendrePolyOrders[nLPOrders - 1] + 1];
  Pl[0] = 1.0;
  block_shft = 0;
  ni = nb * layer[0].nnodes;

  // initialize G_local
  for (int k=0; k<ni; k++) gi[k] = 0.0;

  for (int i=0; i<nb; i++) { // loop over sites i
    if (i) block_shft += 4 * nn[i - 1];
    for (int n=0; n<nSigmas; n++) { // loop over Sigmas
      //for (int m=0; m<5; m++) { // loop over Gis
      for (int j=0; j<nn[i]; j++) { // loop over sites j
        rij = nlist[j * 4 + block_shft];
        if (rij > rc) continue;
        for (int k=0; k<nn[i]; k++) { // loop over sites k
          rik = nlist[k * 4 + block_shft];
          if (rik > rc) continue;
          xj = nlist[j * 4 + 1 + block_shft];
          yj = nlist[j * 4 + 2 + block_shft];
          zj = nlist[j * 4 + 3 + block_shft];
          xk = nlist[k * 4 + 1 + block_shft];
          yk = nlist[k * 4 + 2 + block_shft];
          zk = nlist[k * 4 + 3 + block_shft];
          double costheta = (xj * xk + yj * yk + zj * zk)/(rij * rik);
          tmp = exp(-pow((rij - Sigmas[n]) / SS, 2))*CutoffFunc(rij, rc, Hc);
          if (GiMethod == 0) tmp *= exp(-pow((rik - Sigmas[n]) / SS, 2)) * CutoffFunc(rik, rc, Hc) / 16.0;
          else tmp *= exp(-pow((rik - Sigmas[n]) / SS, 2)) * CutoffFunc(rik, rc, Hc) / pow(Sigmas[n], 2);
          Pl[1] = costheta;
          for (int m=1; m<LegendrePolyOrders[nLPOrders-1]; m++)
            Pl[m + 1] = ((2.0 * m + 1.0) * costheta * Pl[m] - m * Pl[m - 1]) / (m + 1);
          for (int p=0; p<nLPOrders; p++) {
            gi[i * nSigmas * nLPOrders + p * nSigmas + n] += Pl[LegendrePolyOrders[p]] * tmp;
          }
          //gi[i*nSigmas*MAX_LSP + n] += tmp; // Gi(0); m = 0;
          //gi[i*nSigmas*MAX_LSP + nSigmas + n] += Pl[1]*tmp; // Gi(1); m = 1;
          //gi[i*nSigmas*MAX_LSP + 2*nSigmas + n] += Pl[2]*tmp; // Gi(2); m = 2;
          //gi[i*nSigmas*MAX_LSP + 3*nSigmas + n] += Pl[3]*tmp; // Gi(3); m = 3;
          //gi[i*nSigmas*MAX_LSP + 4*nSigmas + n] += Pl[4]*tmp; // Gi(4); m = 4;
          //gi[i*nSigmas*MAX_LSP + 5*nSigmas + n] += Pl[5]*tmp; // Gi(3); m = 5;
          //gi[i*nSigmas*MAX_LSP + 6*nSigmas + n] += Pl[6]*tmp; // Gi(4); m = 6;
        }
      }
      // recondition Gis
      if (gmin != 0.0) {
        for (int p=0; p<nLPOrders; p++) {
          gi[i * nSigmas * nLPOrders + p * nSigmas + n] += gmin;
        }
      }
      switch(GiMethod) {
      case 0:
        break;
      case 1:
        for (int p=0; p<nLPOrders; p++) {
          gi[i * nSigmas * nLPOrders + p * nSigmas + n] = log(gi[i * nSigmas * nLPOrders + p * nSigmas + n]);
        }
        break;
      case 3:
        break;
      case 4:
        for (int p=0; p<nLPOrders; p++) {
          gi[i * nSigmas * nLPOrders + p * nSigmas + n] = log(gi[i * nSigmas * nLPOrders + p * nSigmas + n]);
        }
        break;
      default:
        printf("Error: Gi method:%d not supported .. from file %s at line %d\n",
               GiMethod,__FILE__,__LINE__);
        if (nprocs > 1) MPI_Abort(world,errcode);
        exit(EXIT_FAILURE);
      }
    }
  }

  delete [] Pl;
}

double VolumePerAtom(const double lvec[][3], const int nb)
// compute atomic volume
{
    double t1 = lvec[0][0]*(lvec[1][1]*lvec[2][2] - lvec[2][1]*lvec[1][2]);
    double t2 = lvec[0][1]*(lvec[2][0]*lvec[1][2] - lvec[1][0]*lvec[2][2]);
    double t3 = lvec[0][2]*(lvec[1][0]*lvec[2][1] - lvec[2][0]*lvec[1][1]);
    return fabs((t1 + t2 + t3) / nb);
}

double Area(const VecDoub a, const VecDoub b, VecDoub &axb)
{
  //
  // Returns magnitude and components of vector from cross-product of two vectors.
  //

  double Ax, Ay, Az;

  Ax = a[1] * b[2] - b[1] * a[2];    // ay*bz - by*az, x-component of the cross-product.
  Ay = -(a[0] * b[2] - b[0] * a[2]); // -(ax*bz - bx*az), y-component of the cross-product.
  Az = a[0] * b[1] - b[0] * a[1];    // ax*by - bx*ay, z-component of the cross-product.

  if (axb.size() == 3) {
    axb[0] = Ax;
    axb[1] = Ay;
    axb[2] = Az;
  } else {
    printf("Warning: result of vector cross-product could not be saved .... [%s:%d]\n",
	   __FILE__, __LINE__);
    printf("         size of third argument != 3\n");
  }

  return sqrt(Ax * Ax + Ay * Ay + Az * Az);
}


void One_NeighborList (const double lvec[][3], double **bases, const int nbas, int *&nneighbors, double *&neighbors)
// nneighbors must be of size nbas allocated before passing
// neighbors must be initialized to NULL before passing
// set values of max1, max2 and max3 appropriately before using them.
{
  double x0, y0, z0;
  double x1, y1, z1;
  double r, Rc2;
  int t_nneighbors;
  int tmp, nn;
  int EXTRA = 4;
  double dnorm[3];
  double rc;
  int m1, m2, m3;

  rc = 1.5 * Rc;
  if (PotentialType == 0 || PotentialType == 1) rc = Rc;
  Rc2 = pow(rc,2);

  t_nneighbors = tmp = 0;
  nn = EXTRA;
  neighbors = grow(neighbors, 4*nn,"One_NeighborList()");

  // how far to replicate a supercell
  for (int m=0; m<3; m++) {
    dnorm[m] = sqrt(pow(lvec[m][0], 2) +
        pow(lvec[m][1], 2) + pow(lvec[m][2], 2));
  }
  //std::cout << "I am here ...\n";
  m1 = max1 * (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
  m2 = max2 * (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
  m3 = max3 * (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */
  //printf("structure: %s %d %d %d\n",Lattice[l].name,max1,max2,max3);
  //exit(0);
  for (int k=0; k<nbas; k++) { /* loop over bases */
    for (int i1=-m1; i1<=m1; i1++) {
      for (int i2=-m2; i2<=m2; i2++) {
	for (int i3=-m3; i3<=m3; i3++) {
	  x0 = lvec[0][0]*i1 + lvec[1][0]*i2 + lvec[2][0]*i3;
	  y0 = lvec[0][1]*i1 + lvec[1][1]*i2 + lvec[2][1]*i3;
	  z0 = lvec[0][2]*i1 + lvec[1][2]*i2 + lvec[2][2]*i3;
	  for (int j=0; j<nbas; j++) { /* loop over neighbors */
	    x1 = bases[j][0] + x0 - bases[k][0];
	    y1 = bases[j][1] + y0 - bases[k][1];
	    z1 = bases[j][2] + z0 - bases[k][2];
	    r = x1 * x1 + y1 * y1 + z1 * z1;
	    if (r < Rc2 && r > acc) {
	      r = sqrt(r);
	      nneighbors[k]++;
	      neighbors[t_nneighbors*4] = r;
	      //std::cout << r << "\n";
	      neighbors[t_nneighbors*4 + 1] = x1;
	      neighbors[t_nneighbors*4 + 2] = y1;
	      neighbors[t_nneighbors*4 + 3] = z1;
	      t_nneighbors++;
	      if (t_nneighbors == nn) {
		nn += EXTRA;
		neighbors = grow(neighbors,4*nn,"One_NeighborList()");
	      }
	    }
	  }
	}
      }
    }
  }

  for (int k=0; k<nbas; k++) {
    tmp += nneighbors[k];
  }

  if (tmp != t_nneighbors) {
    errmsg("something wrong in neighbor list calculation",FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  /*for (int i=0; i<t_nneighbors; i++)
	printf("%e %e %e %e\n",neighbors[i*4],neighbors[i*4+1],
		neighbors[i*4+2],neighbors[i*4+3]);*/

  neighbors = grow(neighbors, 4 * t_nneighbors, "One_NeighborList()");
}

void EOS (const char *strucname, const double a, 
	  const double b, const double c)
// This modifies lc_a0, lc_b0, and lc_c0 which are
// global variables.
{
  //int *neighbor_size;
  //double *neighbor_list;
  //int ifound = 0;
  int nb;
  int maxpoints = 2000;
  double trans[3][3], **basis;

  lc_a0 = a;
  lc_b0 = b;
  lc_c0 = c;
  set_struc(strucname, 1.0, nb); // This sets trans0 and basis0.
  //neighbor_list = create(neighbor_list,1,"EOS()");
  //neighbor_size = create(neighbor_size,nb,"EOS()");

  double ds,smin,smax,scale[3];
  double vol, E;
  char fname[256];
  //double *local_Gi;
  FILE *out;

  //local_Gi = NULL;
  sprintf(fname,"EOS_%s",strucname);
  out = fopen(fname,"w");
  basis = create(basis,nb,3,"EOS()");
  smin = 0.8;
  smax = 2.15;
  if (strcmp(strucname, "trimerD3h") == 0 ||
      strcmp(strucname, "trimerC2v") == 0 ||
      strcmp(strucname, "tetramerD4h") == 0 ||
      strcmp(strucname, "pentamerD5h") == 0 ||
      strcmp(strucname, "dimer") == 0 ||
      strcmp(strucname, "tetramerDih") == 0) {
    smin = 0.6;
    smax = 1.8;
  }
  ds = (smax - smin)/maxpoints;
  for (int i=0; i<maxpoints; i++) {
    scale[0] = scale[1] = scale[2] = i*ds + smin;
    if (strcmp(strucname, "dimer") == 0 ||
	strcmp(strucname, "tetramerDih") == 0) scale[1] = scale[2] = 0.0;
    if (strcmp(strucname, "trimerD3h") == 0 ||
	strcmp(strucname, "trimerC2v") == 0 ||
	strcmp(strucname, "tetramerD4h") == 0 ||
	strcmp(strucname, "pentamerD5h") == 0 ||
	strcmp(strucname, "graphitic") == 0) scale[2] = 0.0;
    for (int k=0; k<3; k++) {
      trans[0][k] = trans0[0][k]*scale[k];
      trans[1][k] = trans0[1][k]*scale[k];
      trans[2][k] = trans0[2][k]*scale[k];
      for (int j=0; j<nb; j++) {
	basis[j][k] = basis0[j][k]*scale[k];
      }
    }

    // compute volume per atom
    vol = VolumePerAtom(trans,nb);
    if (strcmp(strucname, "dimer") == 0) vol = fabs(basis[1][0] - basis[0][0]); // Length.
    if (strcmp(strucname, "tetramerDih") == 0) vol = fabs(basis[3][0] - basis[0][0]) / (nb - 1); // Length.
    if (strcmp(strucname, "trimerD3h") == 0 ||
	strcmp(strucname, "trimerC2v") == 0 ||
	strcmp(strucname, "tetramerD4h") == 0 ||
	strcmp(strucname, "pentamerD5h") == 0 ||
	strcmp(strucname, "graphitic") == 0) {
      VecDoub tmp(3);
      VecDoub a(3, trans[0]);
      VecDoub b(3, trans[1]);
      vol = Area(a, b, tmp) / nb; // Area.
    }
    max1 = max2 = max3 = 1;
    //E = crystal_eng(trans,basis,nb);
    E = crystal_eng(trans, basis, nb, strucname);
    fprintf(out,"%f %f\n",vol,E/nb);

    //if (local_Gi) delete [] local_Gi;
  }

  fclose(out);
  //sfree(neighbor_list);
  //sfree(neighbor_size);
  destroy(basis);
  //destroy(basis0);
}


double mean_sqr(const VecDoub &a, const double c)
{
  double sum = 0.0;
  for (int i=0; i<a.size(); i++) sum += a[i]*a[i];
  return c*sum/a.size();
}

double max_filter(const VecDoub &a, const double c)
{
  double fmax;
  fmax = a[0]*a[0];
  for (int i=1; i<a.size(); i++) fmax = MAX(fmax,a[i]*a[i]);
  return c*fmax;
}


double compute_hb_constraint2()
{
  if (PotentialType == 2) {
    // Write local BOP parameters of all atoms.
    if (Nstg == 0) {
      FILE *out;
      out = fopen("hb_BOP.dat","w");
      for (int i=0; i<nprocs; i++) { // loop over blocks.
        for (int j=0; j<num_atoms_per_proc[i]; j++) {// loop over atoms within each block.
          for (int k=0; k<MAX_HB_PARAM; k++) {
            int n = i*max_num_atoms_per_proc*MAX_HB_PARAM + j*MAX_HB_PARAM + k;
            fprintf(out," %e",global_max_pi[n]);
          }
          fprintf(out,"\n");
        }
      }
      fclose(out);
    }

    double sum[MAX_HB_PARAM];
    if (use_filter == 1) {
      for (int i=0; i<MAX_HB_PARAM; i++) sum[i] = 2.0 * __BOP_EST[i];
    } else if (use_filter == 0) {
      for (int i=0; i<MAX_HB_PARAM; i++) sum[i] = 0.0;
      for (int i=0; i<nprocs; i++) { // loop over blocks.
        for (int j=0; j<num_atoms_per_proc[i]; j++) { // loop over atoms within each block.
          for (int k=0; k<MAX_HB_PARAM; k++) { // loop over parameters.
            int n = i*max_num_atoms_per_proc*MAX_HB_PARAM + j*MAX_HB_PARAM + k;
            sum[k] += global_max_pi[n];
          }
        }
      }
      // Average each BO parameter.
      for (int i=0; i<MAX_HB_PARAM; i++) sum[i] /= Ntotal_bases;
    }

    // Compute max. deviation from mean for each column and sum them.
    double sum_errors = 0.0;
    for (int i=0; i<nprocs; i++) { // loop over blocks.
      for (int j=0; j<num_atoms_per_proc[i]; j++) { // loop over atoms within each block.
        for (int k=0; k<MAX_HB_PARAM; k++) { // loop over parameters.
          int n = i * max_num_atoms_per_proc * MAX_HB_PARAM + j * MAX_HB_PARAM + k;
          sum_errors += pow(global_max_pi[n] - sum[k], 2);
        }
      }
    }

    return CONST_HB2 * sum_errors / (Ntotal_bases * MAX_HB_PARAM);
  }

  return 0.0;
}


double crystal_eng(const double trans[][3], double **basis, const int nb)
{
  double *local_Gi = NULL;
  double *nlist;
  int *nnsize;
  double E = 0.0;

  nlist = create(nlist, 1, "crystal_eng:create()");
  nnsize = create(nnsize, nb, "crystal_eng:create()");

  for (int i=0; i<nb; i++) nnsize[i] = 0;

  // Create neighbor list.
  One_NeighborList(trans, basis, nb, nnsize, nlist);

  // Compute energy.
  switch(PotentialType) {
  case 0: // Only BOP
    // compute energy with BOP parameters
    E = Energy(nnsize,nlist,nb,ParamVec);
    break;
  case 1: // Only ANN
    // partition the neural network parameters
    PartitionNNetParams(ParamVec);
    local_Gi = new double [nb * layer[0].nnodes];
    // compute local structural parameters for this state
    iCompute_LSP(nnsize, nlist, nb, local_Gi);
    // compute energy with neural network parameters
    E = NNET_Eng(local_Gi, nb);
    break;
  case 2: // Only PINN
    // partition the neural network parameters
    PartitionNNetParams(ParamVec);
    local_Gi = new double [nb * layer[0].nnodes];
    // compute local structural parameters for this state
    iCompute_LSP(nnsize, nlist, nb, local_Gi);
    // compute energy with neural network parameters
    E = NNET_Eng(local_Gi,nnsize,nlist,nb);
    break;
  default:
    char buf[256];
    sprintf(buf," potential type not found ");
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  if (local_Gi) delete[] local_Gi;
  sfree(nlist);
  sfree(nnsize);

  return E;
}

double crystal_eng(const double trans[][3], double **basis, const int nb, const char *struc)
{
  double *local_Gi = NULL;
  double *nlist;
  int *nnsize;
  double E = 0.0;

  nlist = create(nlist,1,"crystal_eng:create()");
  nnsize = create(nnsize,nb,"crystal_eng:create()");

  for (int i=0; i<nb; i++) nnsize[i] = 0;

  // Create neighbor list.
  One_NeighborList(trans,basis,nb,nnsize,nlist);

  // Compute energy.
  switch(PotentialType) {
  case 0: // Only BOP
    // compute energy with BOP parameters
    E = Energy(nnsize, nlist, nb, ParamVec);
    break;
  case 1: // Only NN
    // partition the neural network parameters
    PartitionNNetParams(ParamVec);
    local_Gi = new double [nb * layer[0].nnodes];
    // compute local structural parameters for this state
    iCompute_LSP(nnsize,nlist,nb,local_Gi);
    // compute energy with neural network parameters
    E = NNET_Eng(local_Gi,nb);
    break;
  case 2: // Only HB
    // partition the neural network parameters
    PartitionNNetParams(ParamVec);
    local_Gi = new double [nb * layer[0].nnodes];
    // compute local structural parameters for this state
    iCompute_LSP(nnsize,nlist,nb,local_Gi);
    // compute energy with neural network parameters
    E = NNET_Eng(local_Gi, nnsize, nlist, nb);
    break;
  default:
    char buf[256];
    sprintf(buf," potential type not found ");
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  // Only for debugging purpose.
#ifdef DEBUG
  if (strcmp(struc,basic_struc) == 0 && PotentialType == 2) {
    FILE *out;
    out = fopen("select_gis_bop.dat","a+");
    double vol = VolumePerAtom(trans,nb);
    if (vol > 16.0 && vol < 100.0) {
      fprintf(out," %f %f",vol,E/nb);
      int size = nSigmas*MAX_LSP;
      for (int i=0; i<size; i++) fprintf(out," %e",local_Gi[i]);
      for (int i=0; i<size; i++) layer[0].vsum[i] = local_Gi[i];
      evaluate_nnet();
      for (int j=0; j<layer[nLayers - 1].nnodes; j++) fprintf(out," %e",layer[nLayers - 1].vsum[j]);
      fprintf(out,"\n");
    }
    fclose(out);
  }
#endif

  if (local_Gi) delete[] local_Gi;
  sfree(nlist);
  sfree(nnsize);

  return E;
}

double crystalFunk(const double x)
{
  int nnb;

  set_struc(basic_struc,x,nnb);

  return crystal_eng(trans0,basis0,nnb);
}

double specific_bop_constraint(VecDoub &pin)
{
  double tmp = 0.0;
  tmp += pow(pin[big_a]-fabs(pin[big_a]),2);
  tmp += pow(pin[alpha]-fabs(pin[alpha]),2);
  tmp += pow(pin[big_b]-fabs(pin[big_b]),2);
  tmp += pow(pin[beta]-fabs(pin[beta]),2);
  //tmp += pow(pin[a]-fabs(pin[a]),2);
  //tmp += pow(pin[lambda]-fabs(pin[lambda]),2);
  return tmp;
}

void compute_LSP()
{
  if (me == 0) printf("Computing Gis .... [%s:%d]\n", __FILE__, __LINE__);

  int n;

  n = layer[0].nnodes;

  for (int i=0; i<NAtoms; i++) {
    atoms[i].Gi_list = create(atoms[i].Gi_list, n, "");
    for (int j=0; j<n; j++) atoms[i].Gi_list[j] = 0.0; // Initialization
    // Only for debugging purpose.
    //for (int j=0; j<atoms[i].nn; j++)
      //printf("rank %d : %f %f %f %f\n", me, atoms[i].nlist[4*j],
	  //atoms[i].nlist[4*j + 1], atoms[i].nlist[4*j + 2], atoms[i].nlist[4*j + 3]);
    //if (me) printf("rank %d: atom %d has %d neighbors\n", me, i, atoms[i].nn);
    compute_atomic_LSP(atoms[i].nn,
		       atoms[i].nlist,
		       atoms[i].Gi_list);
  }
}

void compute_atomic_LSP(const int nn,
			const double *nlist,
			double *gi)
{
  // Computes local structural parameters of an atom.
  //if (me) printf("rank %d entered function: %s\n", me, __FUNCTION__);
  double rij, rik;
  double *Pl;
  double xj, yj, zj;
  double xk, yk, zk;
  double tmp;
  double gmin = REF_GI;
  double rc;

  rc = Rc;

  if (GiMethod == 3 || GiMethod == 4) rc = 1.5 * Rc;

  //if (PotentialType == 0 || PotentialType == 1) rc = Rc;

  Pl = new double [LegendrePolyOrders[nLPOrders - 1] + 1];
  Pl[0] = 1.0;

  for (int n=0; n<nSigmas; n++) { // loop over Sigmas
    //for (int m=0; m<5; m++) {   // loop over Gis
    for (int j=0; j<nn; j++) {    // loop over sites j
      rij = nlist[j * 4];
      if (rij > rc) continue;
      for (int k=0; k<nn; k++) {  // loop over sites k
	rik = nlist[k * 4];
	if (rik > rc) continue;
	xj = nlist[j * 4 + 1];
	yj = nlist[j * 4 + 2];
	zj = nlist[j * 4 + 3];
	xk = nlist[k * 4 + 1];
	yk = nlist[k * 4 + 2];
	zk = nlist[k * 4 + 3];
	double costheta = (xj * xk + yj * yk + zj * zk) / (rij * rik);
	tmp = exp(-pow((rij - Sigmas[n]) / SS, 2)) * CutoffFunc(rij, rc, Hc);
	if (GiMethod == 0) tmp *= exp(-pow((rik - Sigmas[n]) / SS, 2)) * CutoffFunc(rik, rc, Hc) / 16.0;
	else tmp *= exp(-pow((rik - Sigmas[n]) / SS, 2)) * CutoffFunc(rik, rc, Hc) / pow(Sigmas[n], 2);
	Pl[1] = costheta;
	for (int m=1; m<LegendrePolyOrders[nLPOrders-1]; m++)
	  Pl[m + 1] = ((2.0 * m + 1.0) * costheta * Pl[m] - m * Pl[m - 1]) / (m + 1);
	for (int p=0; p<nLPOrders; p++) {
	  gi[p * nSigmas + n] += Pl[LegendrePolyOrders[p]] * tmp;
	}
	//gi[n] += tmp;                       // Gi(0); m = 0;
	//gi[nSigmas + n] += Pl[1] * tmp;     // Gi(1); m = 1;
	//gi[2 * nSigmas + n] += Pl[2] * tmp; // Gi(2); m = 2;
	//gi[3 * nSigmas + n] += Pl[3] * tmp; // Gi(3); m = 3;
	//gi[4 * nSigmas + n] += Pl[4] * tmp; // Gi(4); m = 4;
	//gi[5 * nSigmas + n] += Pl[5] * tmp; // Gi(3); m = 3;
	//gi[6 * nSigmas + n] += Pl[6] * tmp; // Gi(4); m = 4;
      }
    }
    // recondition Gis
    if (gmin != 0.0) {
      for (int p=0; p<nLPOrders; p++) {
	gi[p * nSigmas + n] += gmin;
      }
    }
    switch(GiMethod) {
    case 0:
      break;
    case 1:
      for (int p=0; p<nLPOrders; p++) {
	gi[p * nSigmas + n] = log(gi[p * nSigmas + n]);
      }
      break;
    case 3:
      break;
    case 4:
      for (int p=0; p<nLPOrders; p++) {
	gi[p * nSigmas + n] = log(gi[p * nSigmas + n]);
      }
      break;
    default:
      printf("Error: Gi method:%d not supported .... [%s:%d]\n",
	     GiMethod, __FILE__, __LINE__);
      MPI_Abort(world, EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }
  }

  delete [] Pl;
}
