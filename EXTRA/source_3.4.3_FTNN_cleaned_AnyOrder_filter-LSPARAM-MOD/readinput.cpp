#include <cstring>
#include <math.h> // cmath
#include <unistd.h>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>

#include "globals.h"
#include "readinput.h"
#include "util.h"
#include "mem.h"
#include "compute.h"
#include "defs_consts.h"
#include "nrvector.h"

void CreateNeighborList()
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

  rc = 1.5*Rc;
  if (PotentialType == 0 || PotentialType == 1) rc = Rc;
  Rc2 = pow(rc,2);

  for (int l=0; l<nStruc; l++) { // loop over structures
    t_nneighbors = tmp = 0;
    nn = EXTRA;
    Lattice[l].neighbors = create(Lattice[l].neighbors,4*nn,"CreateNeighborList()");
    // how far to replicate a supercell
    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(Lattice[l].latvec[m][0],2) +
	  pow(Lattice[l].latvec[m][1],2) + pow(Lattice[l].latvec[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */
    //printf("structure: %s %d %d %d\n",Lattice[l].name,max1,max2,max3);
    //exit(0);
    for (int k=0; k<Lattice[l].nbas; k++) { /* loop over bases */
      for (int i1=-m1; i1<=m1; i1++) {
	for (int i2=-m2; i2<=m2; i2++) {
	  for (int i3=-m3; i3<=m3; i3++) {
	    x0 = Lattice[l].latvec[0][0]*i1 + Lattice[l].latvec[1][0]*i2
		+ Lattice[l].latvec[2][0]*i3;
	    y0 = Lattice[l].latvec[0][1]*i1 + Lattice[l].latvec[1][1]*i2
		+ Lattice[l].latvec[2][1]*i3;
	    z0 = Lattice[l].latvec[0][2]*i1 + Lattice[l].latvec[1][2]*i2
		+ Lattice[l].latvec[2][2]*i3;
	    for (int j=0; j<Lattice[l].nbas; j++) { /* loop over neighbors */
	      x1 = Lattice[l].bases[j][0] + x0 - Lattice[l].bases[k][0];
	      y1 = Lattice[l].bases[j][1] + y0 - Lattice[l].bases[k][1];
	      z1 = Lattice[l].bases[j][2] + z0 - Lattice[l].bases[k][2];
	      r = x1 * x1 + y1 * y1 + z1 * z1;
	      if (r < Rc2 && r > acc) {
		r = sqrt(r);
		Lattice[l].nneighbors[k]++;
		Lattice[l].neighbors[t_nneighbors*4] = r;
		//std::cout << r << "\n";
		Lattice[l].neighbors[t_nneighbors*4 + 1] = x1;
		Lattice[l].neighbors[t_nneighbors*4 + 2] = y1;
		Lattice[l].neighbors[t_nneighbors*4 + 3] = z1;
		t_nneighbors++;
		if (t_nneighbors == nn) {
		  nn += EXTRA;
		  Lattice[l].neighbors = grow(Lattice[l].neighbors,4*nn,
					      "CreateNeighborList()");
		}
	      }
	    }
	  }
	}
      }
    }

    for (int k=0; k<Lattice[l].nbas; k++) {
      tmp += Lattice[l].nneighbors[k];
    }

    if (tmp != t_nneighbors) {
      errmsg("something wrong in neighbor list calculation",FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    /*for (int i=0; i<t_nneighbors; i++) printf("%e %e %e %e\n",Lattice[l].neighbors[i*4],
						Lattice[l].neighbors[i*4+1],
						Lattice[l].neighbors[i*4+2],
						Lattice[l].neighbors[i*4+3]);*/
    Lattice[l].total_nneighbors = t_nneighbors;
    Lattice[l].neighbors = grow(Lattice[l].neighbors,4*t_nneighbors,"CreateNeighborList()");
  }
}

int ReadDatabase (const char *file)
// reads structures from input file as an argument to this function call
// and returns number of structures read
{
  int bufsize = 1024;
  int EXTRA_STRUCT = 10;
  char buf[bufsize];
  FILE *in;

  in = fopen(file,"r");

  if (!in) {
    sprintf(buf,"cannot open %s",file);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  int ns = EXTRA_STRUCT;
  Lattice = create(Lattice,ns,"ReadDatabase()");

  int ncount = 0;
  Ntotal_bases = 0;

  while(!feof(in)) {
    fgets(buf,bufsize,in); // structure name
    sscanf(buf,"%s",Lattice[ncount].name);

    //fgets(buf,bufsize,in); // total energy
    //sscanf(buf,"%lf",&Lattice[ncount].E0);

    fgets(buf,bufsize,in); // universal scaling factor

    // lattice vectors
    for (int i=0; i<3; i++) {
      fgets(buf,bufsize,in);
      sscanf(buf,"%lf %lf %lf",&Lattice[ncount].latvec[i][0],
	  &Lattice[ncount].latvec[i][1],&Lattice[ncount].latvec[i][2]);
    }

    fgets(buf,bufsize,in); // number of bases;
    sscanf(buf,"%d",&Lattice[ncount].nbas);
    Ntotal_bases += Lattice[ncount].nbas;

    // compute atomic volume
    Lattice[ncount].omega0 = VolumePerAtom (Lattice[ncount].latvec,Lattice[ncount].nbas);
    Lattice[ncount].G_local = NULL;
    Lattice[ncount].bases = create(Lattice[ncount].bases,
				   Lattice[ncount].nbas,3,"ReadDatabase()");
    Lattice[ncount].nneighbors = create(Lattice[ncount].nneighbors,
					Lattice[ncount].nbas,"ReadDatabase()");
    Lattice[ncount].csort = create(Lattice[ncount].csort,
				   Lattice[ncount].nbas,6,"ReadDatabase()");
    fgets(buf,bufsize,in); // direct or cartesian
    // positions of bases
    for (int i=0; i<Lattice[ncount].nbas; i++) {
      Lattice[ncount].nneighbors[i] = 0;
      fgets(buf,bufsize,in);
      //sscanf(buf,"%lf %lf %lf %s",&Lattice[ncount].bases[i][0],&Lattice[ncount].bases[i][1],
      //&Lattice[ncount].bases[i][2],Lattice[ncount].csort[i]);
      sscanf(buf,"%lf %lf %lf",&Lattice[ncount].bases[i][0],&Lattice[ncount].bases[i][1],
	  &Lattice[ncount].bases[i][2]);
    }

    fgets(buf,bufsize,in); // total energy
    sscanf(buf,"%lf",&Lattice[ncount].E0);
    // Apply shift.
    Lattice[ncount].E0 += shift_E0*Lattice[ncount].nbas;
    strcpy(buf,"\0");
    Lattice[ncount].w = 1.0;
    ncount++;

    if (ncount >= ns) {
      ns += EXTRA_STRUCT;
      //std::cout << "I am here ... ncount = " << ncount << std::endl;
      Lattice = grow(Lattice,ns,"ReadDatabase()");
    }
  }

  fclose(in);
  nStruc = --ncount;
  Lattice = grow(Lattice,ncount,"ReadDatabase()");

  // Assign cluster ids based on their names.
  // First assign same negative integer to each cluster id.
  for (int i=0; i<nStruc; ++i) Lattice[i].cid = -1;
  int start_id = 1;
  for (int i=0; i<nStruc; ++i) {
    if (!i) Lattice[i].cid = start_id;
    else if (Lattice[i].cid < 0) Lattice[i].cid = ++start_id;
    for (int j=i+1; j<nStruc; ++j) {
      if (Lattice[j].cid > 0) continue;
      if (strcmp(Lattice[i].name,Lattice[j].name) == 0) Lattice[j].cid = Lattice[i].cid;
    }
  }
  Ncluster = start_id;

  return ncount;
}

int readinput(const char *progname)
{
  int bufsize=512;
  char buf[bufsize];

  // Check stdin redirection
  if (me == 0) {
    if (isatty(fileno(stdin))) {
      fprintf(stderr,"input file missing!");
      return 1;
    }

    //char ofile[FILENAMESIZE];

    // Process input lines for parameters;
    /*fgets(buf,bufsize,stdin);
    sscanf(buf,"%s",ofile);

    if (strcmp(ofile,"none")==0) logout = stdout;
    else {
      logout = fopen(ofile,"w");
      if (!logout) {
	sprintf(buf,"cannot open %s",ofile);
	errmsg (buf,FERR);
	return 1;
      }
				}*/

    printf("\n");
    printf("Output generated by %s\n",progname);
    printf("Version: %s %s\n\n",__DATE__,__TIME__);

    printf("Input parameters:\n");
    //printf("%s - output file name\n",ofile);

    fgets(buf,bufsize,stdin);
    sscanf(buf,"%s %s", datafile, testfile);
    printf("%s %s - database file\n", datafile, testfile);
    strcpy(testfile, "none"); // test data file is disable for now !!!

    //fgets(buf,bufsize,stdin);
    //sscanf(buf,"%s %s",parfile1,parfile2);
    //printf("%s %s - input and output parameter files for BOP\n",parfile1,parfile2);

    fgets(buf,bufsize,stdin);
    sscanf(buf,"%s %s",NNETParamFileIn,NNETParamFileOut);
    printf("%s %s - input and output parameter files\n",
	   NNETParamFileIn,NNETParamFileOut);

    // read tolerances and annealing schedule (simulated annealing):
    // function tolerance (fTol)
    // gradient tolerance (gTol)
    // start temperature (Tini)
    // end temperature (Tfin)
    // no. of stages or temperatures (Nstg)
    // no. of iterations in each stage/temp (iter0)
    fgets(buf,bufsize,stdin);
    sscanf(buf,"%lf %lf %lf %lf %d %d",&fTol,&gTol,&Tini,&Tfin,&Nstg,&iter0);
    printf("%e %e %e %e %d %d - tolerances and annealing schedule\n",fTol,gTol,Tini,Tfin,Nstg,iter0);

    fgets(buf,bufsize,stdin);
    sscanf(buf,"%d",&PotentialType); // global variable
    printf("%d - Potential type (0:BOP  1:ANN  2:PINN)\n",PotentialType);

    // Amount by which all DFT energies per atom will be shifted.
    fgets(buf,bufsize,stdin);
    sscanf(buf,"%lf",&shift_E0); // global variable.
    printf("%f - amount by which all DFT energies per atom will be shifted.\n",shift_E0);

    // Read name of basic structure and its lattice constants.
    fgets(buf,bufsize,stdin);
    sscanf(buf,"%s %lf %lf %lf",basic_struc,&lc_a0,&lc_b0,&lc_c0); // global variables.
    printf("%s %f %f %f - equil. structure and its lattice constants.\n",basic_struc,lc_a0,lc_b0,lc_c0);
    //sscanf(buf,"%s %lf",basic_struc,&Re0); // global variables.
    //printf("%s %f - equil. structure and its first neighbor distance.\n",basic_struc,Re0);

    // Regularization factors for the constraints.
    fgets(buf,bufsize,stdin);
    sscanf(buf, "%lf %lf %lf %lf", &CONST_BOP, &CONST_NN, &CONST_HB, &CONST_HB2); // global variables.
    // CONST_BOP for pure BOP parameters
    // CONST_NN for NN parameters
    // CONST_HB for mean squares of local BOP parameters of all atoms.
    // CONST_HB2 for regularisation of variance of each local BOP parameter.
    printf("%e %e %e %e - regularization factors for BOP, NN, HB and HB2 constraints.\n",
	   CONST_BOP, CONST_NN, CONST_HB, CONST_HB2);

    // flag to compute Gis or read from a file
    // -1: compute Gis but do not write outputs of the last hidden layer for each stage of iterations.
    //  0: compute Gis and write outputs of the last hidden layer for each stage of iterations.
    //  1: read modified Gis from a given file.
    fgets(buf,bufsize,stdin);
    sscanf(buf,"%d %s",&noGi,GiFile);
    printf("%d %s - flag for Gis settings.\n",noGi,GiFile);

    // if non zero, compute derivatives using finite-difference.
    fgets(buf,bufsize,stdin);
    sscanf(buf,"%d",&isGlobalGrad);
    if (PotentialType == 0) isGlobalGrad = 0; // For straight BOP, use finite difference for now !!!
    printf("%d - if non zero, compute analytical derivatives.\n",isGlobalGrad);

    // read random number seed.
    fgets(buf, bufsize, stdin);
    sscanf(buf, "%ld", &rnd_seed);
    // If zero use current time else use the non-negative integer.
    if (rnd_seed < 0) {
      printf("Error: random number seed cannot be negative .. from file %s at line %d\n",
	     __FILE__,__LINE__);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    printf("%ld - random seed\n", rnd_seed);

    // Read optimization method to use.
    fgets(buf, bufsize, stdin);
    sscanf(buf, "%d", &__OptimizationMethod);
    printf("%d - optimization method (0:DFP 1:GA)\n", __OptimizationMethod);

    // Read parameters of Genetic algorithm.
    if (__OptimizationMethod == 1) {
      // Method of crossing parents and interval to write statistics of the generation.
      fgets(buf, bufsize, stdin);
      sscanf(buf, "%d %d", &__optcross, &__ngwrite);
      printf("%d %d - flag for methods of crossing parents and interval to write statistics of the generation.\n",
	     __optcross, __ngwrite);

      fgets(buf, bufsize, stdin);
      sscanf(buf, "%d %d %d %d", &__np, &__nf, &__ng, &__nmustg);
      printf("%d %d %d %d - number of population, fittest population, generation and mutation stages.\n",
	     __np, __nf, __ng, __nmustg);
      if (__nmustg > 0) {
	__s0 = new double [__nmustg];
	__mustep = new int [__nmustg];
	for (int i=0; i<__nmustg; i++) {
	  fgets(buf, bufsize, stdin);
	  sscanf(buf, "%lf %d", &__s0[i], &__mustep[i]); // mutation stages
	  printf("%f %d - mutation size and steps.\n", __s0[i], __mustep[i]);
	}
      } else {
	if (__nmustg < 0) {
	  printf("Error: number of mutation stages must be > 0 .... [%s:%d]\n",
		 __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
	}
      }
    }
  }

  MPI_Barrier(world);

  // Broadcast globaal variables read by the master;
  MPI_Bcast(element, 3, MPI_CHAR, 0, world);
  MPI_Bcast(&xmass, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(datafile, FILENAMESIZE, MPI_CHAR, 0, world);
  //MPI_Bcast(parfile1,FILENAMESIZE,MPI_CHAR,0,world);
  //MPI_Bcast(parfile2,FILENAMESIZE,MPI_CHAR,0,world);
  MPI_Bcast(NNETParamFileIn, FILENAMESIZE, MPI_CHAR, 0, world);
  MPI_Bcast(NNETParamFileOut, FILENAMESIZE, MPI_CHAR, 0, world);
  MPI_Bcast(&fTol, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&gTol, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Tini, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Tfin, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Nstg, 1, MPI_INT, 0, world);
  MPI_Bcast(&iter0, 1, MPI_INT, 0, world);
  MPI_Bcast(&PotentialType, 1,MPI_INT, 0, world);
  MPI_Bcast(&shift_E0, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&CONST_BOP, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&CONST_NN, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&CONST_HB, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&CONST_HB2, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&noGi, 1, MPI_INT, 0, world);
  MPI_Bcast(GiFile, FILENAMESIZE, MPI_CHAR, 0, world);
  MPI_Bcast(&isGlobalGrad, 1, MPI_INT, 0, world);

  /*MPI_Bcast(&__OptimizationMethod, 1, MPI_INT, 0, world);
  if (__OptimizationMethod == 1) {
    MPI_Bcast(&__optcross, 1, MPI_INT, 0, world);
    MPI_Bcast(&__ngwrite, 1, MPI_INT, 0, wolrd);
    MPI_Bcast(&__np, 1, MPI_INT, 0, world);
    MPI_Bcast(&__nf, 1, MPI_INT, 0, world);
    MPI_Bcast(&__ng, 1, MPI_INT, 0, world);
    MPI_Bcast(&__nmustg, 1, MPI_INT, 0, world);
    if (__nmustg > 0) {
      if (me) {
	__s0 = new double [__nmustg];
	__mustep = new int [__nmustg];
      }
      MPI_Bcast(__s0, __nmustg, MPI_DOUBLE, 0, world);
      MPI_Bcast(__mustep, __nmustg, MPI_INT, 0, world);
    }
  }*/

  MPI_Barrier(world);

  return 0;
}

int ReadBOPParam(const char *file)
{
  int bufsize=512;
  char buf[bufsize];
  FILE *in;

  if (strcmp("none",file) == 0) {
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  // read in BOP fitting parameters;
  in = fopen(file,"r");
  if (!in) {
    sprintf(buf,"cannot open %s",file);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  ParamVec.resize(MAX_BOP_PARAM);
  StepVec.resize(MAX_BOP_PARAM);

  char pstr[bufsize];
  double val, dval;
    
  for (int i=0; i<MAX_BOP_PARAM+1; i++) {
    fgets(buf,bufsize,in);
    sscanf(buf,"%s %lf %lf\n",pstr,&val,&dval);
    if (strcmp(pstr,"A") == 0) {ParamVec[big_a] = val; StepVec[big_a] = dval;}
    else if (strcmp(pstr,"alpha") == 0) {ParamVec[alpha] = val; StepVec[alpha] = dval;}
    else if (strcmp(pstr,"B") == 0) {ParamVec[big_b] = val; StepVec[big_b] = dval;}
    else if (strcmp(pstr,"beta") == 0) {ParamVec[beta] = val; StepVec[beta] = dval;}
    else if (strcmp(pstr,"a") == 0) {ParamVec[small_a] = val; StepVec[small_a] = dval;}
    else if (strcmp(pstr,"h") == 0) {ParamVec[small_h] = val; StepVec[small_h] = dval;}
    else if (strcmp(pstr,"lambda") == 0) {ParamVec[lambda] = val; StepVec[lambda] = dval;}
    else if (strcmp(pstr,"sigma") == 0) {ParamVec[sigma] = val; StepVec[sigma] = dval;}
    //else if (strcmp(pstr,"eta") == 0) {ParamVec[eta] = val; StepVec[eta] = dval;}
    else if (strcmp(pstr,"hc") == 0) {ParamVec[hc] = val; StepVec[hc] = dval;}
    else if (strcmp(pstr,"rc") == 0) {Rc = val;}
  }

  fclose(in);

  return MAX_BOP_PARAM;
}


int ReadData(const char *file, Struc_Data *&data, int &n, int &m)
// This routine reads structures from an input file as first argument
// to this function call and saves in a data structure as second argument.
// Number of bases and clusters are returned as n and m respectively.
// The funciton returns number of structures read.
{
  int bufsize = 1024;
  int EXTRA_STRUCT = 10;
  char buf[bufsize];
  FILE *in;

  in = fopen(file,"r");

  if (!in) {
    sprintf(buf,"cannot open %s",file);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  int ns = EXTRA_STRUCT;
  data = create(data,ns,"ReadDatabase()");

  n = 0;
  int ncount = 0;

  while(!feof(in)) {
    fgets(buf,bufsize,in); // structure name
    sscanf(buf,"%s",data[ncount].name);

    //fgets(buf,bufsize,in); // total energy
    //sscanf(buf,"%lf",&Lattice[ncount].E0);

    fgets(buf,bufsize,in); // universal scaling factor

    // lattice vectors
    for (int i=0; i<3; i++) {
      fgets(buf,bufsize,in);
      sscanf(buf,"%lf %lf %lf",&data[ncount].latvec[i][0],
	  &data[ncount].latvec[i][1],&data[ncount].latvec[i][2]);
    }

    fgets(buf,bufsize,in); // number of bases;
    sscanf(buf,"%d",&data[ncount].nbas);
    n += data[ncount].nbas;

    // compute atomic volume
    data[ncount].omega0 = VolumePerAtom (data[ncount].latvec,data[ncount].nbas);

    data[ncount].bases = create(data[ncount].bases,data[ncount].nbas,3,"ReadDatabase()");
    data[ncount].G_local = NULL;
    data[ncount].nneighbors = create(data[ncount].nneighbors,data[ncount].nbas,"ReadDatabase()");
    data[ncount].csort = create(data[ncount].csort,data[ncount].nbas,6,"ReadDatabase()");
    fgets(buf,bufsize,in); // direct or cartesian
    // positions of bases
    for (int i=0; i<data[ncount].nbas; i++) {
      data[ncount].nneighbors[i] = 0;
      fgets(buf,bufsize,in);
      sscanf(buf,"%lf %lf %lf %s",&data[ncount].bases[i][0],&data[ncount].bases[i][1],
	  &data[ncount].bases[i][2],data[ncount].csort[i]);
    }

    fgets(buf,bufsize,in); // total energy
    sscanf(buf,"%lf",&data[ncount].E0);
    // Apply shift.
    data[ncount].E0 += shift_E0*data[ncount].nbas;
    strcpy(buf,"\0");
    data[ncount].w = 1.0;
    ncount++;

    if (ncount >= ns) {
      ns += EXTRA_STRUCT;
      //std::cout << "I am here ... ncount = " << ncount << std::endl;
      data = grow(data,ns,"ReadDatabase()");
    }
  }

  fclose(in);
  ncount--;
  data = grow(data,ncount,"ReadDatabase()");

  // Assign cluster ids based on their names.
  // First assign same negative integer to each cluster id.
  for (int i=0; i<ncount; ++i) data[i].cid = -1;
  int start_id = 1;
  for (int i=0; i<ncount; ++i) {
    if (!i) data[i].cid = start_id;
    else if (data[i].cid < 0) data[i].cid = ++start_id;
    for (int j=i+1; j<ncount; ++j) {
      if (data[j].cid > 0) continue;
      if (strcmp(data[i].name,data[j].name) == 0) data[j].cid = data[i].cid;
    }
  }

  m = start_id;

  return ncount;
}


void CreateNeighborList(Struc_Data *&data, const int nset)
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

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;
  Rc2 = pow(rc,2);

  for (int l=0; l<nset; l++) { // loop over structures
    t_nneighbors = tmp = 0;
    nn = EXTRA;
    data[l].neighbors = create(data[l].neighbors,4*nn,"CreateNeighborList()");
    // how far to replicate a supercell
    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(data[l].latvec[m][0],2) +
	  pow(data[l].latvec[m][1],2) + pow(data[l].latvec[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */
    //printf("structure: %s %d %d %d\n",Lattice[l].name,max1,max2,max3);
    //exit(0);
    for (int k=0; k<data[l].nbas; k++) { /* loop over bases */
      for (int i1=-m1; i1<=m1; i1++) {
	for (int i2=-m2; i2<=m2; i2++) {
	  for (int i3=-m3; i3<=m3; i3++) {
	    x0 = data[l].latvec[0][0]*i1 + data[l].latvec[1][0]*i2
		+ data[l].latvec[2][0]*i3;
	    y0 = data[l].latvec[0][1]*i1 + data[l].latvec[1][1]*i2
		+ data[l].latvec[2][1]*i3;
	    z0 = data[l].latvec[0][2]*i1 + data[l].latvec[1][2]*i2
		+ data[l].latvec[2][2]*i3;
	    for (int j=0; j<data[l].nbas; j++) { /* loop over neighbors */
	      x1 = data[l].bases[j][0] + x0 - data[l].bases[k][0];
	      y1 = data[l].bases[j][1] + y0 - data[l].bases[k][1];
	      z1 = data[l].bases[j][2] + z0 - data[l].bases[k][2];
	      r = x1 * x1 + y1 * y1 + z1 * z1;
	      if (r < Rc2 && r > acc) {
		r = sqrt(r);
		data[l].nneighbors[k]++;
		data[l].neighbors[t_nneighbors*4] = r;
		//std::cout << r << "\n";
		data[l].neighbors[t_nneighbors*4 + 1] = x1;
		data[l].neighbors[t_nneighbors*4 + 2] = y1;
		data[l].neighbors[t_nneighbors*4 + 3] = z1;
		t_nneighbors++;
		if (t_nneighbors == nn) {
		  nn += EXTRA;
		  data[l].neighbors = grow(data[l].neighbors,4*nn,
					      "CreateNeighborList()");
		}
	      }
	    }
	  }
	}
      }
    }

    for (int k=0; k<data[l].nbas; k++) {
      tmp += data[l].nneighbors[k];
    }

    if (tmp != t_nneighbors) {
      errmsg("something wrong in neighbor list calculation",FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    /*for (int i=0; i<t_nneighbors; i++) printf("%e %e %e %e\n",Lattice[l].neighbors[i*4],
						Lattice[l].neighbors[i*4+1],
						Lattice[l].neighbors[i*4+2],
						Lattice[l].neighbors[i*4+3]);*/
    data[l].total_nneighbors = t_nneighbors;
    data[l].neighbors = grow(data[l].neighbors,4*t_nneighbors,"CreateNeighborList()");
  }
}

void read_Modified_Gis(char *fname)
{
  std::ifstream in;
  in.open(fname);
  if (!in) {
    char buf[256];
    sprintf(buf," cannot open file: %s ",fname);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,1);
    exit(EXIT_FAILURE);
  }

  // get length of file;
  in.seekg(0, std::ios::end);
  int length = in.tellg();
  in.seekg(0, std::ios::beg);

  // allocate memory;
  char *buffer = new char [length];

  // read the file as a block;
  in.read(buffer, length);
  in.close();

  std::string file;
  file.assign(buffer);
  std::stringstream ss;
  ss.str(file);

  delete [] buffer;

  std::string longline; // for reading a line
  std::stringstream iss; // for reading each word in a line
  std::string word;

  for (int i=0; i<nStruc; i++) {
    Lattice[i].G_local = create(Lattice[i].G_local,Lattice[i].nbas*layer[0].nnodes,
	"read_Modified_Gis:create()");
    for (int j=0; j<Lattice[i].nbas; j++) {
      getline(ss,longline);
      iss.str(longline);
      // skip first two fields
      iss >> word;
      iss >> word;
      for (int k=0; k<layer[0].nnodes; k++) {
	iss >> word;
	Lattice[i].G_local[j*layer[0].nnodes + k] = atof(word.c_str());
      }
      iss.clear();
    }
  }
}

void read_Modified_Gis_cpus(char *fname)
{
  //
  // This subroutine must be called after
  // information of atoms and their neighbor
  // lists are distributed to all cpus.
  //

  //printf("Rank %d reading saved Gis .... [%s:%d]\n", me, __FILE__, __LINE__);
  std::ifstream in;
  in.open(fname);

  if (!in) {
    printf("Error: cannot open file: %s .... [%s:%d]\n",
	   fname, __FILE__, __LINE__);
    MPI_Abort(world, MPI_ERR_NO_SUCH_FILE);
    exit(EXIT_FAILURE);
  }

  // get length of file;
  in.seekg(0, std::ios::end);
  int length = in.tellg();
  in.seekg(0, std::ios::beg);

  // allocate memory;
  char *buffer = new char [length];

  // read the file as a block;
  in.read(buffer, length);
  in.close();

  std::string file;
  file.assign(buffer);
  std::stringstream ss;
  ss.str(file);

  delete [] buffer;

  std::string longline; // for reading a line
  std::stringstream iss; // for reading each word in a line
  std::string word;

  int gi_size = layer[0].nnodes;
  int count = 0;


  // Get its share of Gis.
  while (count < NAtoms && ss.good()) {
    getline(ss, longline);
    iss.str(longline);
    iss >> word;
    int idx = atoi(word.c_str()); // atom id
    if (atoms[count].atomid == idx) {
      if (atoms[count].Gi_list == NULL)
	atoms[count].Gi_list = create(atoms[count].Gi_list,
				      gi_size,
				      "read_Modified_Gis:create()");
      else atoms[count].Gi_list = grow(atoms[count].Gi_list,
					 gi_size,
					 "read_Modified_Gis:create()");
      // skip next three words.
      iss >> word;
      iss >> word;
      for (int j=0; j<gi_size; j++) {
	iss >> word;
	atoms[count].Gi_list[j] = atof(word.c_str());
      }
      count++;
    }
    iss.clear();
  }

  // Root also saves all Gis for future use.
  if (me == 0) {
    if (!__GiList) {
      printf("Error: the structure not allocated .... [%s:%d]\n", __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_OTHER);
      exit(0);
    }
    printf("Rank 0 reading all Gis .... [%s:%d]\n", __FILE__, __LINE__);
    // Set input position to begining of the stringstream.
    ss.seekg(0, ss.beg);
    for (int i=0; i<Ntotal_bases; i++) {
      getline(ss, longline);
      iss.str(longline);
      iss >> word;
      int idx = atoi(word.c_str());
      __GiList[i].atomid = idx;     // atom id
      iss >> word;
      int gdx = atoi(word.c_str()); // group/configuration id
      __GiList[idx].gid = gdx;
      iss >> word;                  // cluster name; no need to save
      for (int j=0; j<gi_size; j++) {
	iss >> word;
	__GiList[idx].gilist[j] = atof(word.c_str());
      }
      iss.clear();
    }
  }
}
