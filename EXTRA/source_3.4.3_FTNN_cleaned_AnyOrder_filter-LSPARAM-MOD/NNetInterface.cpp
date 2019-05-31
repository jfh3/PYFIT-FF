#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "globals.h"
#include "NNetInterface.h"
#include "nrmatrix.h"
#include "util.h"
#include "mem.h"
#include "compute.h"
#include "ran.h"
#include "write.h"

void evaluate_nnet()
{
  int i, j;

  for (i=1; i<nLayers-1; i++) {
    // y <= Ax.
    cblas_dgemv(CblasRowMajor,CblasTrans,layer[i-1].nnodes,layer[i].nnodes,
	1.0,layer[i].Weights,layer[i].nnodes,layer[i-1].vsum,1,0.0,layer[i].mvprod,1);
    // y <= a + b.
    // Specific to Intel MKL library.
    //vdAdd(layer[i].nnodes,layer[i].mvprod,layer[i].Biases,layer[i].vsum);
    //y = ax + y.
    cblas_daxpy(layer[i].nnodes,1.0,layer[i].Biases,1,layer[i].mvprod,1);
    cblas_dcopy(layer[i].nnodes,layer[i].mvprod,1,layer[i].vsum,1);
    // apply logistic function to a vector.
    switch(LogisticFuncType) {
    case 0: // sigmoid
      for (j=0; j<layer[i].nnodes; j++) {
	layer[i].mvprod[j] = 1.0/(1.0 + exp(-layer[i].vsum[j]));
	if (isLocalGrad) layer[i].fdot[j] = layer[i].mvprod[j] * (1.0 - layer[i].mvprod[j]);
      }
      break;
    case 1: // 1/2 tanh(x/2)
      for (j=0; j<layer[i].nnodes; j++) {
	layer[i].mvprod[j] = 1.0/(1.0 + exp(-layer[i].vsum[j])) - 0.5;
	if (isLocalGrad) layer[i].fdot[j] = 0.25 - layer[i].mvprod[j]*layer[i].mvprod[j];
      }
      break;
    default:
      printf("Error: logistic function type: %d not supported .... [%s:%d]\n",
	     LogisticFuncType, __FILE__, __LINE__);
      if (nprocs > 1) MPI_Abort(world,1);
      exit(EXIT_FAILURE);
    }
    // copy mvprod to vsum.
    cblas_dcopy(layer[i].nnodes,layer[i].mvprod,1,layer[i].vsum,1);
  }
  // output layer does not apply logistic function.
  cblas_dgemv(CblasRowMajor,CblasTrans,layer[i-1].nnodes,layer[i].nnodes,
      1.0,layer[i].Weights,layer[i].nnodes,layer[i-1].vsum,1,0.0,layer[i].mvprod,1);
  //vdAdd(layer[i].nnodes,layer[i].mvprod,layer[i].Biases,layer[i].vsum);
  cblas_daxpy(layer[i].nnodes,1.0,layer[i].Biases,1,layer[i].mvprod,1);
  cblas_dcopy(layer[i].nnodes,layer[i].mvprod,1,layer[i].vsum,1);
}

void PartitionNNetParams(VecDoub &p)
// Vector of neural network parameters are assigned to 
// weights and biases of hiden and output layers in that order.
{
  int i = 0;
  // The first layer is the input layer and skipped.
  // Node values of input layer are structural
  // properties of atoms.
  for (int l=1; l<nLayers; l++) {
    // weights: m x n matrix
    for (int m=0; m<layer[l-1].nnodes; m++) { // rows
      for (int n=0; n<layer[l].nnodes; n++) { // cols
	layer[l].Weights[m*layer[l].nnodes+n] = p[i++];
      }
    }
    // biases: vector of size nnodes.
    for (int m=0; m<layer[l].nnodes; m++) { // cols
      layer[l].Biases[m] = p[i++];
    }
  }

  if (i != p.size()) {
    printf("Error: read %d instead of %d parameters .. from file %s at line %d\n",
	   i,p.size(),__FILE__,__LINE__);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }
}

void ReadNNetParams(const char *file, VecDoub &p,
		    VecDoub &dy, double &s_total)
// This routine reads in neural network configuration
// and allocates storage for an array of layer.
// It also allocates memory for matrices in each layer.
{
  int bufsize = 1024;
  char buf[bufsize];
  FILE *in;

  // Only master reads in values of the fitting parameters;
  if (me == 0) {
    if (strcmp("none",file) == 0) {
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    in = fopen(file,"r");
    if (!in) {
      sprintf(buf,"cannot open %s",file);
      errmsg(buf,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    // read version number and reference Gi.
    fgets(buf,bufsize,in);
    sscanf(buf,"%d %lf %d", &GiMethod, &REF_GI, &LogisticFuncType);

    // read number of chemical species in the system.
    fgets(buf,bufsize,in);
    sscanf(buf,"%d",&nChemSort);
    // read nChemSort names and massess.
    for (int i=0; i<nChemSort; i++) {
      fgets(buf,bufsize,in);
      sscanf(buf, "%s %lf", element[i], &xmass[i]);
    }

    // read flag to choose random initialization of
    // all weights and biases or read in from a given file
    fgets(buf,bufsize,in);
    sscanf(buf, "%d %lf %lf %lf %lf", &NNetInit, &MAX_RANGE, &Rc, &Hc, &SS);
    // if NNetInit==0, read weights and biases else randomly assign weights and biases.

    if (MAX_RANGE == 0.0 || Rc <= 0.0 || Hc <= 0.0) {
      sprintf(buf,"something wrong with range, cutoff distance, or cutoff range: (max_range:%f Rc:%f Hc:%f) !\n",
	      MAX_RANGE, Rc, Hc);
      errmsg(buf,FERR);
      fclose(in);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    int nwords;
    char **values;
    char *next;
    char emsg[bufsize];

    //
    // Read number of Legendre-Polynomial orders and
    // the orders in accending order.
    // example: [5] [0 1 2 4 6]
    //
    fgets(buf,bufsize,in);
    next = strchr(buf,'\n');
    if (next) {
      *next = '\0';
      nwords = count_words(buf);
      values = new char* [nwords];
      values[0] = strtok(buf," \t\n\r\f");
      nLPOrders = atoi(values[0]);
      if (nLPOrders <= 0) {
	printf("Error: must be positive non zero integer .... [%s:%d]\n",
	       __FILE__,__LINE__);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      if (nLPOrders) {
	LegendrePolyOrders = new (std::nothrow) int [nLPOrders];
	//mem_usage += nSigmas * sizeof(double);
	if (LegendrePolyOrders == nullptr) {
	  printf("Error: failed to allocate memory .... in file %s at line %d\n",
		 __FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
	  exit(EXIT_FAILURE);
	}
	for (int i=1; i<=nLPOrders; i++) {
	  values[i] = strtok(NULL," \t\n\r\f");
	  LegendrePolyOrders[i-1] = atoi(values[i]);
	}
      }
    }
    else {
      printf("Error: blank line found .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_OTHER);
      exit(EXIT_FAILURE);
    }
    delete [] values;

    //
    // Read number of Sigmas and values.
    //
    fgets(buf,bufsize,in);
    next = strchr(buf,'\n');
    if (next) {
      *next = '\0';
      nwords = count_words(buf);
      values = new char* [nwords];
      values[0] = strtok(buf," \t\n\r\f");
      nSigmas = atoi(values[0]);
      if (nSigmas != nwords-1) {
	sprintf(emsg,"incorrect number of entries: (%d != %d)!!!\n",
		nSigmas, nwords-1);
	errmsg(emsg,FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      if (nSigmas) {
	Sigmas = new double [nSigmas];
	if (Sigmas == NULL) {
	  printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
	  exit(EXIT_FAILURE);
	}
	for (int i=1; i<=nSigmas; i++) {
	  values[i] = strtok(NULL," \t\n\r\f");
	  Sigmas[i-1] = atof(values[i]);
	}
      }
    }
    else {
      errmsg("blank line",FERR);
      if (nprocs > 1) MPI_Abort(world,1);
      exit(EXIT_FAILURE);
    }
    delete [] values;

    //
    // Read a flag to use filters.
    //
    __BOP_EST = new double [MAX_HB_PARAM];
    fgets(buf, bufsize, in);
    sscanf(buf, "%d %lf %lf %lf %lf %lf %lf %lf %lf", &use_filter,
	   &__BOP_EST[big_a],
	   &__BOP_EST[alpha],
	   &__BOP_EST[big_b],
	   &__BOP_EST[beta],
	   &__BOP_EST[small_h],
	   &__BOP_EST[sigma],
	   &__BOP_EST[small_a],
	   &__BOP_EST[lambda]);
    /*printf("%d %f %f %f %f %f %f %f %f - flag and estimates of local BO parameters.\n",
	   use_filter,
	   __BOP_EST[big_a],
	   __BOP_EST[alpha],
	   __BOP_EST[big_b],
	   __BOP_EST[beta],
	   __BOP_EST[small_a],
	   __BOP_EST[small_h],
	   __BOP_EST[lambda],
	   __BOP_EST[sigma]);*/

    //
    // Read number of layers and
    // number of nodes in each layer
    // for each interaction.
    //
    fgets(buf, bufsize, in);
    next = strchr(buf,'\n');
    if (next) {
      *next = '\0';
      nwords = count_words(buf);
      values = new char* [nwords];
      values[0] = strtok(buf," \t\n\r\f");
      nLayers = atoi(values[0]);
      if (nLayers != nwords-1) {
	sprintf(emsg,"incorrect number of entries: (%d != %d)!!!\n",
		nLayers, nwords-1);
	errmsg(emsg,FERR);
	fclose(in);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      if (nLayers > 0 && nLayers < 3) {
	sprintf(emsg,"there must be at least one hidden layer and one output layer: total(%d) including input layer!!!\n",
		nLayers);
	errmsg(emsg,FERR);
	fclose(in);
	if (nprocs > 1) MPI_Abort(world,1);
	exit(EXIT_FAILURE);
      }
      if (nLayers) {
	//layer = (int*) malloc(nLayers*sizeof(int));
	layer = new Layer [nLayers];
	if (layer == NULL) {
	  printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
	  exit(EXIT_FAILURE);
	}
	for (int i=1; i<=nLayers; i++) {
	  values[i] = strtok(NULL," \t\n\r\f");
	  layer[i-1].nnodes = atoi(values[i]);
	}
      }
    } else {
      errmsg("blank line",FERR);
      fclose(in);
      if (nprocs > 1) MPI_Abort(world,1);
      exit(EXIT_FAILURE);
    }
    delete [] values;
    // only for debugging purpose
    //printf("Rank(%d) %d",me,nLayers[l]);
    //for (int i=0; i<nLayers; i++) printf(" %d",layer[i].nnodes);
    //printf("\n");
    //exit(0);
  }

  MPI_Barrier(world);
  MPI_Bcast(&GiMethod, 1, MPI_INT, 0, world);
  MPI_Bcast(&REF_GI, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&LogisticFuncType, 1, MPI_INT, 0, world);
  MPI_Bcast(&NNetInit, 1, MPI_INT, 0, world);
  MPI_Bcast(&MAX_RANGE, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Rc, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Hc, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&SS, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&nLPOrders, 1, MPI_INT, 0, world);
  if (me) LegendrePolyOrders = new int [nLPOrders];
  MPI_Bcast(LegendrePolyOrders, nLPOrders, MPI_INT, 0, world);
  MPI_Bcast(&nSigmas, 1, MPI_INT, 0, world);
  if (me) {
    Sigmas = new double [nSigmas];
    if (Sigmas == NULL) {
      printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
      if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }
  }
  MPI_Bcast(&use_filter, 1, MPI_INT, 0, world);
  if (use_filter) {
    if (me && PotentialType == 2) __BOP_EST = new double [MAX_HB_PARAM];
    if (PotentialType == 2) MPI_Bcast(__BOP_EST, MAX_HB_PARAM, MPI_DOUBLE, 0, world);
  }
  MPI_Bcast(Sigmas, nSigmas, MPI_DOUBLE, 0, world);
  MPI_Bcast(&nLayers,1,MPI_INT,0,world);
  if (me) {
    layer = new Layer [nLayers];
    if (layer == NULL) {
      printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
      if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }
  }

  MPI_Barrier(world);

  for (int i=0; i<nLayers; i++) MPI_Bcast(&layer[i].nnodes,1,MPI_INT,0,world);
  if (NNetInit == 1) {
    if (layer[0].nnodes != (nSigmas * nLPOrders)) {
      if (me == 0) printf("Warning: number of input Gis updated to %d .... [%s:%d]\n",
			  (nSigmas * nLPOrders), __FILE__, __LINE__);
      layer[0].nnodes = nSigmas * nLPOrders;
    }
  }
  //printf("rank %d: no. of nodes = %d\n",me,layer[0].nnodes);
  //MPI_Bcast(layer,nLayers,MPI_INT,0,world);

  // compute total number of neural network parameters
  nNNPARAM = 0; // Rc and Hc are not included
  for (int i=1; i<nLayers; i++) {
    nNNPARAM += layer[i].nnodes + layer[i-1].nnodes * layer[i].nnodes;
  }
  //printf("Rank(%d): total number of NNets parameters = %d\n",me,nNNPARAM);
  //exit(0);
  MPI_Barrier(world);

  // allocate memory for neural network parameters.
  if (nLayers) {
    int indx_last = nLayers - 1; // index of output layer.
    for (int i=0; i<nLayers; i++) {
      if (!i) {
	layer[i].vsum = (double *)malloc(layer[i].nnodes*sizeof(double));
	if (layer[i].vsum == NULL) {
	  printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
	  exit(EXIT_FAILURE);
	}
      }
      else {
	layer[i].Weights = (double *)malloc(layer[i-1].nnodes*layer[i].nnodes*sizeof(double));
	layer[i].Biases = (double *)malloc(layer[i].nnodes*sizeof(double));
	layer[i].mvprod = (double *)malloc(layer[i].nnodes*sizeof(double));
	layer[i].vsum = (double *)malloc(layer[i].nnodes*sizeof(double));
	if (layer[i].Weights == NULL || layer[i].Biases == NULL || layer[i].mvprod == NULL ||
	    layer[i].vsum == NULL) {
	  printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
	  if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
	  exit(EXIT_FAILURE);
	}
	if (i != indx_last) {
	  layer[i].fdot = (double *)malloc(layer[i].nnodes*sizeof(double));
	  layer[i].SMat = (double *)malloc(layer[i].nnodes*layer[indx_last].nnodes*sizeof(double));
	  if (layer[i].fdot == NULL || layer[i].SMat == NULL) {
	    printf("Error: failed to allocate memory .... in file %s at line %d\n",__FILE__,__LINE__);
	    if (nprocs > 1) MPI_Abort(world,EXIT_FAILURE);
	    exit(EXIT_FAILURE);
	  }
	}
      }
    }
  }

  p.resize(nNNPARAM); // include global parameters ALPHA, BETA and LAMBDA.
  dy.resize(nNNPARAM);

  if (me == 0) {
    // ==== read remaining parameters ====
    if (NNetInit == 0) {
      //Ranq1 ran(time(NULL));
      for (int i=0; i<nNNPARAM; i++) {
	fgets(buf,bufsize,in);
	sscanf(buf,"%lf %lf",&p[i],&dy[i]);
	// Perturb each parameter for specific optimizations.
	if (dy[i] > 0.0) p[i] += p[i]*dy[i];

	s_total += dy[i];
      }
      // ---- end of reading neural network parameters ----
    } else {
      // ---- randomly fill neural network parameters ----
      long int seed_value;
      if (rnd_seed == 0) seed_value = time(NULL);
      else seed_value = rnd_seed;
      Ranq1 ran(seed_value);

      double tmpval;
      for (int i=0; i<nNNPARAM; i++) {
	//do {
	tmpval = MAX_RANGE * (2.0 * ran.doub() - 1.0);
	//} while (fabs(tmpval) > MAX_RANGE || fabs(tmpval) < 0.1);
	p[i] = tmpval;
	dy[i] = 0.0;
	s_total += dy[i];
      }
      //p[nNNPARAM-1] = Hc; //MAX_RANGE * (2.0 * rand() / RAND_MAX - 1.0);
    }
    fclose(in);
  }

  // broadcast dy[];
  double *tmp_dy = new double [dy.size()];
  for (int i=0; i<dy.size(); i++) tmp_dy[i] = dy[i];
  MPI_Bcast(&tmp_dy[0],dy.size(),MPI_DOUBLE,0,world);
  if (me) {
    for (int i=0; i<dy.size(); i++) dy[i] = tmp_dy[i];
  }
  delete [] tmp_dy;

  // Wait everybody;
  MPI_Barrier(world);
}

void NNetOutput (double &f0, const int m)
// Only for NNET
{
		evaluate_nnet();
  f0 = layer[nLayers-1].vsum[0];
}

void NNetOutput(const int *nn, const double *nlist,
		const int Basis_Id, double &f0, const int m)
// For NN or PINN
// m is reserved for later use.
{
  evaluate_nnet();
  if (PotentialType == 1) f0 = layer[nLayers-1].vsum[0]; // only NN
  else if (PotentialType == 2) {       // PINN
    int n = layer[nLayers-1].nnodes;
    VecDoub pvec(n);
    // 'a' must be a single row vector.
    // Assign content of 'a' to pvec.
    if (use_filter) for (int j=0; j<n; j++)
      pvec[j] = layer[nLayers-1].vsum[j] + __BOP_EST[j];
    else {
      for (int j=0; j<n; j++)
        pvec[j] = layer[nLayers-1].vsum[j];
    }

    f0 = Atomic_Eng(nn, nlist, Basis_Id, pvec);
  }
}


double NNET_Eng(const double *lsparam, const int nbas)
{
  //
  // -----------------------------------------
  // only for NN.
  // Weights and biases are known to all cpus.
  //------------------------------------------
  //

  int size = layer[0].nnodes;

  double f0, E = 0.0;

  for (int i=0; i<nbas; i++) {
    // update vsum of layer[0]
    for (int j=0; j<size; j++) layer[0].vsum[j] = lsparam[i*size + j];
    NNetOutput(f0,0);
    //for (int j=0; j<size; j++) printf(" %e",layer[0].vsum[j]);
    //printf(" %e\n",f0);
    E += f0;
  }

  return E;
}

double NNET_Eng(const double *lsparam, const int Struc_Id, const int nbas)
// Serial or 1 cpu version.
// Lattices are known.
// Sizes of weights and biases are known.
{
  int size;

  if (noGi == 1) size = layer[0].nnodes;
  else size = nSigmas*MAX_LSP;

  if (size != layer[0].nnodes) {
    errmsg("no. of input biases does not match with no. of local structural parameters",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  double f0, E = 0.0;

  for (int i=0; i<nbas; i++) {
    // update vsum of layer[0]
    for (int j=0; j<size; j++) layer[0].vsum[j] = lsparam[i*size + j];
    switch(PotentialType) {
    case 1: // Only for NN
      NNetOutput(NULL,NULL,i,f0,0);
      break;
    case 2: // For PINN
      NNetOutput(Lattice[Struc_Id].nneighbors,Lattice[Struc_Id].neighbors,i,f0,0);
      break;
    default:
      errmsg("potential type currently not supported",FERR);
      if (nprocs > 1) MPI_Abort(world,1);
      exit(EXIT_FAILURE);
      break;
    }
    E += f0;
  }

  return E;
}

double NNET_Eng(const double *lsparam, const int *nn,
		const double *nlist, const int nbas)
// for PINN.
{
  int size = layer[0].nnodes;

  double f0, E = 0.0;

  for (int i=0; i<nbas; i++) {
    // update vsum of layer[0]
    for (int j=0; j<size; j++) layer[0].vsum[j] = lsparam[i*size + j];
    if (PotentialType == 1) NNetOutput(NULL,NULL,i,f0,0); // Pure neural network.
    else if (PotentialType == 2) {
      NNetOutput(nn,nlist,i,f0,0); // Mix - neural network and BOP
    } else {
      errmsg("problem in no. of nodes at the output layer of the neural network",FERR);
      if (nprocs > 1) MPI_Abort(world,1);
      exit(EXIT_FAILURE);
    }
    E += f0;
  }

  return E;
}


double iNNET_Eng(const double *lsparam, const double *nlist, const int nn)
// Only for Hybrid (NNET + BOP).
{
  int size;

  if (noGi == 1) size = layer[0].nnodes;
  else size = nSigmas*MAX_LSP;

  if (size != layer[0].nnodes) {
    errmsg("no. of input biases does not match with no. of local structural parameters",FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  // update biases of layer[0]
  for (int j=0; j<size; j++) layer[0].vsum[j] = lsparam[j];

  //double begin = MPI_Wtime();
  evaluate_nnet();
  //printf(" Rank %d completed multiplication of weights and biases in %e s.\n",
	 //me,MPI_Wtime()-begin);

  //if (Nstg == 0 && PotentialType == 2)
    //printf(" %e %e %e %e %e %e %e %e\n",a(0,0),a(0,1),a(0,2),a(0,3),a(0,4),a(0,5),a(0,6),a(0,7));
  int n = layer[nLayers-1].nnodes;
  VecDoub pvec(n);
  for (int j=0; j<n; j++) pvec[j] = layer[nLayers-1].vsum[j];

  HBconstraint += specific_bop_constraint(pvec);
  HBconstraint += mean_sqr(pvec,CONST_HB); // constraint for local BOP parameters.

  double f0 = iEnergy(nn,nlist,pvec);

  return f0;
}

//#include <sys/time.h>
double iNNET_Eng(const double *lsparam, const double *nlist, const int nn, VecDoub &pvec)
// Only for PINN potential.
{
  int size = layer[0].nnodes;

  // update vsum of layer[0]
  for (int j=0; j<size; j++) layer[0].vsum[j] = lsparam[j];

  evaluate_nnet();

  int n = layer[nLayers-1].nnodes;
  if (n != MAX_HB_PARAM) {
    printf("Error: size mismatch from file %s at line %d\n",
           __FILE__, __LINE__);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  if (use_filter)
    for (int j=0; j<n; j++) pvec[j] = layer[nLayers-1].vsum[j] + __BOP_EST[j];
  else
    for (int j=0; j<n; j++) pvec[j] = layer[nLayers-1].vsum[j]; // Add the estimates to the local BO parameters.
  //std::cout << pvec;

  double f0 = iEnergy(nn, nlist, pvec);

  return f0;
}
