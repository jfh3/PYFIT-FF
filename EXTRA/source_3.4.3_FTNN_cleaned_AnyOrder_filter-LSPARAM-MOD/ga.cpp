#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <errno.h>
#include <cstring>

#include "write.h"
#include "globals.h"
#include "ga.h"
#include "ran.h"
#include "memory.h"
#include "util.h"
#include "compute.h"
#include "NNetInterface.h"
#include "sorting_algo.h"
#include "mpi_stuff.h"

void GeneticAlgo(VecDoub &p, const VecDoub &ds)
{
  if (me == 0) {
    int npar;
    if (PotentialType == 0) npar = MAX_BOP_PARAM;
    if (PotentialType == 1 || PotentialType == 2) npar = nNNPARAM;
    double **pop,*d,**fittest;
    double dev;
    int m,nmusum,j1,j2,j3;
    int ngene,*mgene_grid,mgene;
    int nnf,nselect,ncount;
    double ss,dd1,sum,sum2;
    double ave,stdev;
    double StartTime;
    double TotalSortingTime,StartSortingTime;
    double num,deno;
    double *actual_dev;

    // clear NNetInit
    NNetInit = 0;

    StartTime = MPI_Wtime();
    TotalSortingTime = 0.0;
    d = NULL;
    pop = fittest = NULL;
    mgene_grid = NULL;

    mgene_grid = new int[npar];
    d = create(d, __np,"");
    actual_dev = create(actual_dev, __np, "");
    fittest = create(fittest, __nf, npar, "GeneticAlgo():create()");
    pop = create(pop, __np, npar, "GeneticAlgo:create()");

    printf("Generating %d population each of %d parameters ....\n", __np, npar);
    fflush(stdout);

    // Start random number generator and initialize.
    long int seed_value;
    if (rnd_seed <= 0) seed_value = time(NULL);
    else seed_value = rnd_seed;
    printf("Random seed to generate initial population: %ld\n", seed_value);
    Ranq1 ran(seed_value);

    // Generate initial population
    for (int i=0; i<__np; i++) {
      //printf("%d ==>",i);
      for (int j=0; j<npar; j++) {
	double dum;
	if (PotentialType == 0) {
	  if (i) dum = 2.0*ran.doub() - 1.0;
	  else dum = 0.0;
	  pop[i][j] = p[j] + ds[j]*dum*p[j];
	} else if (PotentialType == 1 || PotentialType == 2) {
	  if (NNetInit) {
	    if (i) pop[i][j] = MAX_RANGE * (2.0*ran.doub() - 1.0);
	    else pop[i][j] = p[j];
	  } else {
	    if (i) dum = 2.0*ran.doub() - 1.0;
	    else dum = 0.0;
	    pop[i][j] = p[j] + ds[j]*dum*p[j];
	  }
	}
	//printf(" %f",pop[i][j]);
      }
      //printf("\n");
    }

    // Process the parameters of the initial population
    printf("Computing fitnesses of the initial population ... Wait ...");

    for (int i=0; i<__np; i++) {
      for (int j=0; j<npar; j++) p[j] = pop[i][j];
      d[i] = Funk(p); // p is still a global variable, ParamVec.
      actual_dev[i] = sqrt(d[i] - (BOPconstraint + NNconstraint + HBconstraint + HBconstraint2));
    }

    printf("Sorting the population first time based on their fitnesses ... Wait ... ");

    // Sort the set of parameters in ascending order of goodness of fit
    StartSortingTime = MPI_Wtime();
    NewSort(d, pop, __np, npar, 1);
    TotalSortingTime += (MPI_Wtime() - StartSortingTime) / __np;
    //printf(" It took %f seconds to complete sorting %d sets.\n",ttime,np);

    for (int i=0; i<__nf; i++) {
      for (int j=0; j<npar; j++) fittest[i][j] = pop[i][j];
    }

    printf("Creating siblings and writing outputs to a file ...\n");

    printf("# GEN\tBESTDEV\tMEANDEV\tSTDDEV\tTIME(s)\tSIZE\n");
    //TotalTime += MPI_Wtime() - StartTime;
    actual_dev[0] = d[0] - (BOPconstraint + NNconstraint + HBconstraint + HBconstraint2);
    printf("%8d%17.7e%17.7e%17.7e %.2f %d\n",
	   0, actual_dev[0], 0.0, 0.0, MPI_Wtime() - StartTime, __np);

    FILE *best;
    best = fopen("best.param","w");
    if (!best) {
      printf("Error: cannot open file `best.param' .... [%s:%d]\n",
	     __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_OTHER);
      exit(1);
    }

    for (int igen=1; igen<=__ng; igen++) {
      StartTime = MPI_Wtime();
      // Resize d and fittest;
      d = grow(d, __nf, "");
      fittest = grow(fittest, __nf, npar, "");
      if (igen==1) {
	m = 0;
	ss = __s0[m];
	nmusum = __mustep[m];
	printf("# mutation stage: %d %f\n", m, ss);
	//iseed = time(NULL);
      } else {
	if (m<__nmustg) {
	  if ((igen%nmusum)==1) {
	    m += 1;
	    ss = __s0[m];
	    nmusum += __mustep[m];
	    printf("# mutation stage: %d %f\n", m, ss);
	    //iseed = time(NULL);
	  }
	}
      }

      // Reproduction/Crossing
      for (int i=__nf; i<__np; i++) {
	j1 = (int) floor(__nf * ran.doub());
	//j2 = (int) floor(nf*ran.doub());
	do {
	  j2 = (int) floor(__nf * ran.doub());
	} while (j1 == j2);
	switch (__optcross) {
	// Crossing schemes.
	case 0:
	  for (int j=0; j<npar; j++) {
	    int sign = (int) round(ran.doub());
	    pop[i][j] = fittest[j1 * sign + (1 - sign) * j2][j];
	  }
	  break;
	case 1: dd1 = d[j1] + d[j2]; //log(d[j1]*d[j2]);
	  for (int j=0; j<npar; j++) {
	    pop[i][j] = fittest[j2][j] * d[j1] / dd1 + fittest[j1][j] * d[j2] / dd1;
	  }
	  break;
	case 2:
	  do {
	    j3 = (int) floor(__nf * ran.doub());
	  } while (j3 == j1 || j3 == j2);
	  for (int j=0; j<npar; j++) {
	    num = (pow(fittest[j2][j], 2) - pow(fittest[j3][j], 2)) * log(d[j1])
		+ (pow(fittest[j3][j], 2) - pow(fittest[j1][j], 2)) * log(d[j2])
		+ (pow(fittest[j1][j], 2) - pow(fittest[j2][j], 2)) * log(d[j3]);
	    deno = (fittest[j2][j] - fittest[j3][j]) * log(d[j1])
		+ (fittest[j3][j] - fittest[j1][j]) * log(d[j2])
		+ (fittest[j1][j] - fittest[j2][j]) * log(d[j3]);
	    pop[i][j] = 0.5 * num / deno;
	  }
	  break;
	default:
	  for (int j=0; j<npar; j++) {
	    int sign = (int) round(ran.doub());
	    pop[i][j] = fittest[j1 * sign + (1 - sign) * j2][j];
	  }
	  break;
	}

	// Mutation
	// How many genes to mutate;
	ngene = (int) round(npar * ran.doub());
	if (ngene > 0 && ngene < npar) {
	  for (int l=0; l<npar; l++) mgene_grid[l] = 0;
	  for (int k=0; k<ngene; k++) {
	    do {
	      // Which gene to pick;
	      mgene = (int) floor(npar*ran.doub());
	    } while (mgene_grid[mgene]!=0 && ds[mgene]==0.0);
	    mgene_grid[mgene] = 1;
	    double dum = 2.0 * ran.doub() - 1.0;
	    pop[i][mgene] = pop[i][mgene] * (1.0 + dum * ss);
	  }
	} else if (ngene == npar){
	  for (int k=0; k<ngene; k++) {
	    double dum = 2.0 * ran.doub() - 1.0;
	    pop[i][k] = pop[i][k] * (1.0 + dum * ss);
	  }
	}

	// Only for debugging purpose;
	//for (int k=0; k<npar/2; k++) printf("%18.10e%18.10e%18.10e%18.10e\n",pop[i][npar/2+k],pop[i][k],ds[npar/2+k],ds[k]);
	//printf("\n");
      }
      // End of reproduction //
        
      // Compute fitness;
      nselect = 0;
      ncount = __nf;
      for (int i=__nf; i<__np; i++) {
	for (int j=0; j<npar; j++) p[j] = pop[i][j];
	// Only for debugging purpose;
	//for (int k=0; k<npar/2; k++) printf("%18.10e%18.10e%18.10e%18.10e\n",pop[i][npar/2+k],pop[i][k],ds[npar/2+k],ds[k]);
	//printf("\n");
	dev = Funk(p);
	if (dev < d[__nf - 1]) {
	  nselect++;
	  if (nselect > ncount - __nf) {
	    ncount += EXTRA;
	    d = grow(d, ncount, "");
	    fittest = grow(fittest, ncount, npar, "");
	  }
	  d[nselect + __nf - 1] = dev;
	  for (int j=0; j<npar; j++) fittest[nselect + __nf - 1][j] = pop[i][j];
	}
      }
      nnf = nselect + __nf;

      //printf("nnf = %d\n",nnf);
      d = grow(d,nnf,"GeneticAlgo()");
      fittest = grow(fittest, nnf, npar, "GeneticAlgo");

      /*if ((igen%ngwrite)==0) {
	for (int i=0; i<np; i++) {
          for (int j=0; j<npar; j++) fprintf(hist,"%17.8e",pop[i][j]);
          fprintf(hist,"\n");
        }
        fprintf(hist,"\n\n");
        fflush(hist);
                        }*/

      StartSortingTime = MPI_Wtime();
      NewSort(d, fittest, nnf, npar, 1);
      TotalSortingTime += (MPI_Wtime() - StartSortingTime) / nnf;
      
      // Only for debugging purpose;
      /*for (int i=0; i<3; i++) {
          printf("%17.8e",d[i]);
        for (int j=0; j<npar; j++) printf("%17.8e",fittest[i][j]);
        printf("\n");
      }*/
      
      // Report the best set of parameters
      sum = 0.0;
      sum2 = 0.0;
      for (int i=0; i<__nf; i++) {
	actual_dev[i] = sqrt(d[i] - (BOPconstraint + NNconstraint + HBconstraint + HBconstraint2));
	sum += actual_dev[i];
	sum2 += actual_dev[i] * actual_dev[i];
      }
      ave = sum / __nf;
      stdev = sqrt(fabs(sum2 / __nf - (sum / __nf) * (sum / __nf)));
      //TotalTime += MPI_Wtime() - StartTime;
      //printf("error: %s\n",strerror(errno));
      printf("%8d%17.7e%17.7e%17.7e %.2f %d\n",
	     igen, actual_dev[0], ave, stdev, MPI_Wtime() - StartTime, nnf);

      for (int j=0; j<npar; j++) p[j] = fittest[0][j]; // p must be the global, ParamVec.
      rewind(best);
      if (PotentialType == 0) WriteBOPParam(best);
      if (PotentialType == 1 || PotentialType == 2) WriteNNetParam(best);
    }
    // ===== End of ng generations loop ===== //

    // Copy the fittest parameter set to a global array.
    for (int j=0; j<npar; j++) p[j] = fittest[0][j];

    // Close all opened file streams.
    if (best) fclose(best);

    printf("THE BEST DEVIATION AFTER GENERATION %d : %17.7e\n\n", __ng, actual_dev[0]);
    printf("AVERAGE TIME FOR SORTING PER SET: %.8e s\n", TotalSortingTime / (__ng + 1));

    // Delete memory allocations
    sfree(d);
    sfree(actual_dev);
    delete[] mgene_grid;
    destroy(pop);
    destroy(fittest);

    // Send zero MPI_TAG to other processes to return them to the main program;
    //double *dummy = new double [npar];
    //for (int i=1; i<nprocs; i++) MPI_Send(dummy,npar,MPI_DOUBLE,i,0,world);
    //delete [] dummy;

    // Compute structural energies associated with the best set of parameters.
    double tmp = Funk(p);
    // send dummy data to all slave nodes with tag 0 here.
    mpi_send_dummy();

    // END OF THE MASTER PART   
  } else {
    mpi_exchange_slaves_nnet();
  }
}
