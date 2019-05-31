#include <iostream>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <ctime>

#include "globals.h"
#include "write.h"
#include "analytical.h"
#include "readinput.h"
#include "ran.h"
#include "mem.h"
#include "util.h"
#include "NNetInterface.h"
#include "defs_consts.h"
#include "compute.h"
#include "mpi_stuff.h"
#include "search.h"
#include "crystal_struc.h"
#include "surfaces.h"
#include "elastic.h"
#include "ga.h"

int main(int argc, char **argv)
{
  double starttime, endtime;

  // Initialize MPI
  errcode = MPI_Init(&argc, &argv);
  world = MPI_COMM_WORLD;

  if (errcode != MPI_SUCCESS) {
    errmsg("fail to start MPI program",FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  // Get number of processes called
  MPI_Comm_size(world,&nprocs);
  // Get my rank
  MPI_Comm_rank(world,&me);

  // Record start time
  if (me == 0) starttime = MPI_Wtime();

  // Report number of processors.
  if (me == 0) {
    std::cout << "Notes on this version:\n";
    std::cout << " 1. Weights and biases of a neural network are \n";
    std::cout << "    randomly selected in the range 0.0 < |wi,bi| < MAX_RANGE. \n";
    std::cout << " 2. All Gis may be shifted by REF_GI.\n";
    std::cout << " 3. In routine, dfpmin() STPMX = 100.0\n";
    std::cout << " 4. Reads in number of orders of Legendre polynomials and the orders.\n";
    std::cout << " 5. dfpmin() uses gTol. Choose appropriate gTol.\n";
    std::cout << " 6. For PINN potential, cutoff distance and range are held constant.\n";
    std::cout << " 7. Local BOP parameters, a and lambda are squared.\n";
    std::cout << " 8. Options to choose between regular Gis and natural logarithm of Gis \n";
    std::cout << "    after a given shift.\n";
    std::cout << " 9. Options to choose between sigmoid and 1/2tanh(x/2) functions.\n";
    std::cout << "10. For PINN, outputs from ANN are shifted by baseline values \n";
    std::cout << "    which are obtained from straight BOP fit.\n";
    std::cout << "Optimization algorithms:\n";
    std::cout << "  DFP_minimize() Davidson-Fletcher-Powell quasi Newton\n";
    std::cout << " strain step, epsilon = " << epsilon << "\n";
    std::cout << " Number of processors used: " << nprocs << "\n";
  }

  double stot;
  nelast = 5; //21;
  stot = 0.0;
  Lattice = NULL;
  TestSet = NULL;
  atoms = NULL;
  layer = NULL;
  Sigmas = NULL;
  acc = 1.0e-6;

  // Let the master read the input options
  errcode = readinput(argv[0]);

  if (errcode != MPI_SUCCESS) {
    errmsg("reading input parameters",FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  if (me == 0) {
    if (PotentialType == 0) {
      int np = ReadBOPParam(NNETParamFileIn);
      #ifdef DEBUG
      // only for debugging purpose
      std::cout << "======= For straight BOP parameters =======\n";
      std::cout << "number of fitting parameters: " << np << "\n";
      std::cout << ParamVec;
      std::cout << StepVec;
      std::cout << "===========================================\n";
      #endif
    }
  }

  // It is important to broadcast Rc to all cpus
  // since it is fixed and not included in the
  // BOP parameter set.
  if (PotentialType == 0) MPI_Bcast(&Rc,1,MPI_DOUBLE,0,world);

  if (PotentialType == 1 || PotentialType == 2) {
    // Read neural network parameters in ParamVec.
    // Parameters require proper partitioning later.
    ReadNNetParams(NNETParamFileIn,ParamVec,StepVec,stot);
  }

  if (me == 0) { // Report number of fitting parameters for a potential type
    switch (PotentialType) {
    case 0: printf("Number of fitting parameters: %d\n",MAX_BOP_PARAM);
      break;
    case 1: printf("Number of fitting parameters: %d\n",nNNPARAM);
      break;
    case 2: printf("Number of fitting parameters: %d\n",nNNPARAM);
      break;
    }
  }

  // Check output layer nodes.
  switch (PotentialType) {
  case 1:
    if (layer[nLayers - 1].nnodes != 1) {
      fprintf(stderr,"Number of nodes in the output layer does not match .... from file %s at line %d\n",
	      __FILE__,__LINE__);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    break;
  case 2:
    if (layer[nLayers - 1].nnodes != 8) {
      errmsg("`number of nodes in the output layer does not match'",FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    break;
  }

  // Only the root reads atomic structures from database
  // and make neighbor list for each atom in each structure
  if (me == 0) {
    ReadDatabase(datafile);
    printf("Number of configurations: %d\n", nStruc);
    //fflush(logout);
    if (nStruc == 0) {
      errmsg("reading data file failed: check data format\n",FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    printf("Number of clusters: %d\n", Ncluster);
    printf("Total number of bases: %d\n", Ntotal_bases);

    double stime;
    stime = MPI_Wtime();
    CreateNeighborList();
    printf("It took %f secs to create neighborlists in %d configurations.\n",
	   MPI_Wtime()-stime,nStruc);
  }

  MPI_Bcast(&nStruc, 1, MPI_INT, 0, world); // total number of configurations
  MPI_Bcast(&Ntotal_bases, 1, MPI_INT, 0, world); // total number of atoms

  // Distribute atoms.
  mpi_dist_data3();

  // Only for debugging purpose.
  /*printf("rank %d has %d atoms.\n", me, NAtoms);
  for (int i=0; i<NAtoms; i++)
    for (int j=0; j<atoms[i].nn; j++)
      printf("%d %f %f %f %f\n", atoms[i].nn,
	     atoms[i].nlist[4*j],
	  atoms[i].nlist[4*j + 1],
	  atoms[i].nlist[4*j + 2],
	  atoms[i].nlist[4*j + 3]);

  MPI_Barrier(world);
  exit(0);*/

  // Compute Gis or read from the given file.
  //double stime;
  if (me == 0 ) {
    if (PotentialType == 1 || PotentialType == 2) {
      __GiList = new LSPARAMETER [Ntotal_bases];
      for (int i=0; i<Ntotal_bases; i++)
	__GiList[i].gilist = new double [layer[0].nnodes];
    }
  }

  switch(noGi) {
  case -1:
  case 0:
    if (PotentialType == 1 || PotentialType == 2) {
      //stime = MPI_Wtime();
      compute_LSP();
      //printf("Rank %d finished computing Gis of %d atoms in %f(secs)\n",
      //       me, NAtoms, MPI_Wtime() - stime);
      MPI_Barrier(world);
      // Write local-structural parameters of all atoms.
      WriteLSP_mpi(); // requires __Gilist.
    }
    break;
  case 1:
    // do not compute Gis; use outputs from the last hidden layer.
    // read saved outputs from the specified file
    if (PotentialType == 1 || PotentialType == 2) {
      //stime = MPI_Wtime();
      if (me == 0) printf("Reading local structural parameters from a given file ....\n");
      read_Modified_Gis_cpus(GiFile);
      //printf("Rank %d finished reading Gis of %d atoms in %f(secs)\n",
      //       me, NAtoms, MPI_Wtime() - stime);
      //WriteLSP();
    }
    break;
  default:
    printf("Error: unsupported value of noGi is not properly set .... [%s:%d]\n",
	   __FILE__, __LINE__);
    MPI_Abort(world, MPI_ERR_OTHER);
    exit(EXIT_FAILURE);
    break;
  }

  if (me == 0) {
    if (PotentialType == 1 || PotentialType == 2) WriteMatrixFormat(stdout);

#ifdef DEBUG
    // only for debugging purpose
    std::cout << "======== Information of configurations =======\n";
    for (int i=0; i<nStruc; i++) {
      printf("%s\n",Lattice[i].name);
      printf("total energy: %.8e\n",Lattice[i].E0);
      printf("%.8e %.8e %.8e\n",Lattice[i].latvec[0][0],
	  Lattice[i].latvec[0][1],Lattice[i].latvec[0][2]);
      printf("%.8e %.8e %.8e\n",Lattice[i].latvec[1][0],
	  Lattice[i].latvec[1][1],Lattice[i].latvec[1][2]);
      printf("%.8e %.8e %.8e\n",Lattice[i].latvec[2][0],
	  Lattice[i].latvec[2][1],Lattice[i].latvec[2][2]);
      printf("atomic volume: %.8e\n",Lattice[i].omega0);
      printf("%d\n",Lattice[i].nbas);
      for (int j=0; j<Lattice[i].nbas; j++) {
	printf("%.8e %.8e %.8e %s\n",Lattice[i].bases[j][0],Lattice[i].bases[j][1],
	    Lattice[i].bases[j][2],Lattice[i].csort[j]);
      }
    }

    std::cout << "==== number of neighbors of each atom in each structure ====\n";
    for (int i=0; i<nStruc; i++) {
      std::cout << Lattice[i].name << " " << Lattice[i].total_nneighbors << ":\n";
      for (int j=0; j<Lattice[i].nbas; j++) {
	std::cout << " " << Lattice[i].nneighbors[j];
      }
      std::cout << "\n";
    }

    std::cout << "==== neighbor distance and positions in first structure ====\n";
    int k = 0;
    printf("total no. of neighbors in first structure = %d\n",Lattice[0].total_nneighbors);
    for (int j=0; j<Lattice[41].nbas; j++) {
      if (j) k += 4*Lattice[41].nneighbors[j-1];
      printf(" %d",j+1);
      for (int i=0; i<Lattice[41].nneighbors[j]; i++)
	printf(" %d %.8e %.8e %.8e %.8e\n",i+1,Lattice[0].neighbors[i*4+k],
	    Lattice[0].neighbors[i*4+1+k],Lattice[0].neighbors[i*4+2+k],
	    Lattice[0].neighbors[i*4+3+k]);
      printf("\n");
    }
    // Write max and min distances.
    WritMaxMinDist("NeighborDist.dat");
#endif

    // Read test data.
    if (strcmp(testfile,"none") != 0) {
      nTestSize = ReadData(testfile,TestSet,nTestBases,nTestCluster);
      printf(" Number of configurations in the test set: %d\n",nTestSize);
      printf(" Number of bases in the test set: %d\n",nTestBases);
      printf(" Number of groups in the test set: %d\n",nTestCluster);

#ifdef DEBUG
      // only for debugging purpose
      std::cout << "Debugging ....\n";
      printf(" Number of configurations in the test set: %d\n",nTestSize);
      for (int i=0; i<nTestSize; i++) {
	printf("%s\n",TestSet[i].name);
	printf("%.8e %.8e %.8e\n",TestSet[i].latvec[0][0],
	    TestSet[i].latvec[0][1],TestSet[i].latvec[0][2]);
	printf("%.8e %.8e %.8e\n",TestSet[i].latvec[1][0],
	    TestSet[i].latvec[1][1],TestSet[i].latvec[1][2]);
	printf("%.8e %.8e %.8e\n",TestSet[i].latvec[2][0],
	    TestSet[i].latvec[2][1],TestSet[i].latvec[2][2]);
	printf("atomic volume: %.8e\n",TestSet[i].omega0);
	printf("%d\n",TestSet[i].nbas);
	for (int j=0; j<TestSet[i].nbas; j++) {
	  printf("%.8e %.8e %.8e\n",TestSet[i].bases[j][0],TestSet[i].bases[j][1],
	      TestSet[i].bases[j][2]);
	}
	printf("total energy: %.8e\n",TestSet[i].E0);
      }
#endif

      // Compute neighbor list of the test set.
      CreateNeighborList(TestSet,nTestSize);
#ifdef DEBUG
      for (int i=0; i<TestSet[0].total_nneighbors; i++)
	printf("%e %e %e %e\n",TestSet[0].neighbors[i*4],
	    TestSet[0].neighbors[i*4+1],
	    TestSet[0].neighbors[i*4+2],
	    TestSet[0].neighbors[i*4+3]);
#endif

      // Compute local-structural parameters of the test set.
      ComputeLocalStrucParam(TestSet,nTestSize);
    }

    if (Ntotal_bases < nprocs -1) {
      printf("No. of atoms (%d) is less than no. of cores (%d) ...\n",
	     Ntotal_bases,nprocs-1);
      printf("No efficiency gain !!!\n");
    }
  }

  // All cpus may not get same number of atoms because of rounding in
  // integer division, Ntotal_bases/nStruc;
  // find the maximum number of atoms per processor and assign to max_num_atoms_per_proc.
  MPI_Allreduce(&NAtoms, &max_num_atoms_per_proc, 1, MPI_INT, MPI_MAX, world);

  if (PotentialType == 2) {
    local_max_pi = new double [max_num_atoms_per_proc*MAX_HB_PARAM];
    if (local_max_pi == NULL) {
      printf("Error: failed to allocate memory .... [%s:%d]\n",
	     __FILE__,__LINE__);
      MPI_Abort(world,EXIT_FAILURE);
    }
  }
  if (me == 0 && PotentialType == 2) {
    global_max_pi = new double [nprocs*max_num_atoms_per_proc*MAX_HB_PARAM];
    if (global_max_pi == NULL) {
      printf("Error: failed to allocate memory .... [%s:%d]\n",
	     __FILE__,__LINE__);
      MPI_Abort(world,EXIT_FAILURE);
    }
  }

  // Allocate memory for local and global arrays.
  local_eng_sum = new double[nStruc+1]; // 1 extra for HBconstraint.
  if (local_eng_sum == NULL) {
    printf("Error: failed to allocate memory .... [%s:%d]\n",
	   __FILE__,__LINE__);
    MPI_Abort(world,EXIT_FAILURE);
  }

  if (PotentialType == 2 && isGlobalGrad) {
    local_sum_partial_deriv = new double [nStruc*nNNPARAM];
    local_sum_hbconst_partial_deriv = new double [nNNPARAM];
    local_partial_deriv = new double [max_num_atoms_per_proc*MAX_HB_PARAM*nNNPARAM];
    if (local_sum_partial_deriv == NULL || local_sum_hbconst_partial_deriv == NULL
	|| local_partial_deriv == NULL) {
      printf("Error: failed to allocate memory .... [%s:%d]\n",
	     __FILE__,__LINE__);
      MPI_Abort(world,EXIT_FAILURE);
    }
  }

  if (me == 0) {
    // 1 extra for HBconstraint;
    // only relevant to the root.
    global_eng_sum = new double[nStruc+1];
    if (global_eng_sum == NULL) {
      printf("Error: failed to allocate memory .... [%s:%d]\n",
	     __FILE__,__LINE__);
      MPI_Abort(world,EXIT_FAILURE);
    }
    if (PotentialType == 2 && isGlobalGrad) {
      global_sum_partial_deriv = new double [nStruc*nNNPARAM];
      global_sum_hbconst_partial_deriv = new double [nNNPARAM];
      global_partial_deriv = new double [nprocs*max_num_atoms_per_proc*MAX_HB_PARAM*nNNPARAM];
      if (global_sum_partial_deriv == NULL || global_sum_hbconst_partial_deriv == NULL ||
	  global_partial_deriv == NULL) {
 printf("Error: failed to allocate memory .... [%s:%d]\n",
	       __FILE__,__LINE__);
	MPI_Abort(world,EXIT_FAILURE);
      }
    }
  }

  // num_atoms_per_proc:
  //  ______ ______ _______ ____ _____________
  // |rank 0|rank 1|rank 2|----|rank nprocs-1|
  //  ------ ------ ------ ---- -------------
  if (me == 0) {
    num_atoms_per_proc = new int [nprocs];
    if (num_atoms_per_proc == NULL) {
      printf("Error: failed to allocate memory .... [%s:%d]\n",
	     __FILE__,__LINE__);
      MPI_Abort(world,EXIT_FAILURE);
    }
  }
  // collect NAtoms from each processor and store in num_atoms_per_proc in rank 0.
  MPI_Gather(&NAtoms, 1, MPI_INT, num_atoms_per_proc, 1, MPI_INT, 0, world);
  //printf(" Rank %d received %d atoms and max. no. of atoms = %d.\n",me,NAtoms,max_num_atoms_per_proc);

  MPI_Barrier(world);

#ifdef DEBUG
  if (me == 0) {
    printf(" Rank %d keeps distribution of number of atoms in each processor:\n",me);
    for (int i=0; i<nprocs; i++) printf("cpu %d has %d atoms ...\n",i,num_atoms_per_proc[i]);
    printf("\n");
  }
#endif

  // Report initial values of fitting parameters.
  if (me == 0) {
    for (int i=0; i<nChemSort; i++) {
      printf("ELEMENT: %s MASS: %f\n",element[i],xmass[i]);
      //fflush(logout);
    }
    // Print initial values of fitting parameters.
    printf("INITIAL VALUES OF FITTING PARAMATERS:\n");
    if (PotentialType == 0) {
      WriteBOPParam(stdout);
      printf("Input parameter file:   %s\n", NNETParamFileIn);
      printf("Output parameter file:  %s\n", NNETParamFileOut);
      //fflush(logout);
    } else {
      if (PotentialType == 1 || PotentialType == 2) {
        WriteNNetParam(stdout);
        printf("Input parameter file:   %s\n", NNETParamFileIn);
        printf("Output parameter file:  %s\n", NNETParamFileOut);
        //fflush(logout);
      }
    }
  }

  //if (me == 0) printf("I am here .... [%s:%d]\n", __FILE__, __LINE__);
  // Wait everybody;
  MPI_Barrier(world);

  // Optimize the fitting parameters.
  switch(__OptimizationMethod) {
  case 0:
    DFP_minimize();
    break;
  case 1:
    GeneticAlgo(ParamVec, StepVec);
    break;
  default:
    printf("Error: optimization method not supported .... [%s:%d]\n",
	   __FILE__, __LINE__);
    MPI_Abort(world, MPI_ERR_OTHER);
    exit(0);
  }

  // Make sure that neural network parameters
  // are read from the saved file next time.
  if (NNetInit) NNetInit = 0;

  Nstg = 1;

  // ============================================
  // print optimized values of fitting parameters
  // ============================================
  if (me == 0) {
    FILE *paramout;
    char buf[1024];
    printf("OPTIMIZED VALUES OF FITTING PARAMETERS:\n");
    if (PotentialType == 0) {
      WriteBOPParam(stdout);
      paramout = fopen(NNETParamFileOut, "w");
      if (!paramout) {
        sprintf(buf, "cannot open file %s", NNETParamFileOut);
        errmsg(buf, FERR);
        if (nprocs > 1) MPI_Abort(world,1);
        exit(EXIT_FAILURE);
      }
      WriteBOPParam(paramout);
      fclose(paramout);
    }

    if (PotentialType == 1 || PotentialType == 2) {
      WriteNNetParam(stdout);
      paramout = fopen(NNETParamFileOut,"w");
      if (!paramout) {
	sprintf(buf,"cannot open file %s",NNETParamFileOut);
	errmsg(buf,FERR);
	if (nprocs > 1) MPI_Abort(world,1);
	exit(EXIT_FAILURE);
      }
      WriteNNetParam(paramout);
      fclose(paramout);
    }

    //  compare the desired and optimized physical properties
    WriteProperty();

    if (noGi != 1) {
      // Find equilibrium energy of the basic structure.
      max1 = max2 = max3 = 1;
      double x = fmin(0.95,1.05,crystalFunk,1.e-6);
      int nnb;
      set_struc(basic_struc, x, nnb);
      E0 = crystal_eng(trans0, basis0, nnb) / nnb;
      printf(" deformation = %.8f\n",x);
      printf(" equil. energy of %s structure = %.8f (eV/atom)\n",basic_struc,E0);
      omega0 = VolumePerAtom(trans0,nnb);
      printf(" equil. volume of %s structure = %f (A^3/atom)\n",
	      basic_struc,omega0);

      // Compute energy of low index surfaces.
      lc_a0 *= x;
      lc_b0 *= x; // fitted equilibrium lattice constants.
      lc_c0 *= x;

      surf100();
      surf110();
      surf111();


      // Compute formation energy of a vacancy.
      if (set_vac_supercell(basic_struc,1.0,nnb)) {
	double vac_eng = crystal_eng(trans0,basis0,nnb);// - nnb*E0;
	printf(" vacancy formation energy = %.8f (eV) (%.8e) \n",vac_eng-nnb*E0,vac_eng);
      }

      // Compute formation energy of a tetrahedral interstitial.
      if (set_Td_int(basic_struc,1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" Td interstitial formation energy = %f (eV)\n",int_eng);
      }

      // Compute formation energy of a Octahedral interstitial.
      if (set_Octa_int(basic_struc,1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" Octahedral interstitial formation energy = %f (eV)\n",int_eng);
      }

      // Compute formation energy of a dumbbell-100 interstitial.
      if (set_dumbbell_int(basic_struc,"100",1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" <100> dumbbell formation energy = %f (eV)\n",int_eng);
      }

      // Compute formation energy of a dumbbell-110 interstitial.
      if (set_dumbbell_int(basic_struc,"110",1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" <110> dumbbell formation energy = %f (eV)\n",int_eng);
      }

      // Compute formation energy of a dumbbell-111 interstitial.
      if (set_dumbbell_int(basic_struc,"111",1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" <111> dumbbell formation energy = %f (eV)\n",int_eng);
      }

      // Compute formation energy of a HEX interstitial.
      if (set_HEX_int(basic_struc,1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" HEX interstitial formation energy = %f (eV)\n",int_eng);
      }

      // Compute formation energy of a bond-center interstitial.
      if (set_B_int(basic_struc,1.0,nnb)) {
	double int_eng = crystal_eng(trans0,basis0,nnb) - nnb*E0;
	printf(" bond-center interstitial formation energy = %f (eV)\n",int_eng);
      }

      // Compute bulk modulus.
      printf(" bulk modulus = %f (GPa)\n",BulkModulus());
      double C11 = c11();
      printf(" c11 = %f (GPa)\n",C11);
      double shear_mod = ShearModulus();
      printf(" shear modulus = %f (GPa)\n",shear_mod);
      printf(" c12 = %f (GPa)\n",C11-shear_mod);
      printf(" c44 = %f (GPa)\n",c44());

      // Compute GSF energies.
      SFp111();
      SFp100();
      SFp110();

      /*
      // compute EOSs
      EOS("fcc",4.05,0.0,0.0);
      EOS ("bcc",3.31,0.0,0.0);
      EOS ("diam",6.61,0.0,0.0);
      EOS("sc",2.9,0.0,0.0);
      EOS("A15",5.8,5.8,5.8);
      EOS("hex",2.9,0.0,0.0);
      // provide a and c that
      // match c/a ratio from DFT data.
      EOS("hcp",2.9,0.0,4.77917);
      EOS("dimer",2.9,0.0,0.0);
      EOS("trimerD3h",2.9,0.0,0.0);
      EOS("trimerC2v",2.9,0.0,0.0);
      EOS("tetramerDih",2.9,0.0,0.0);
      EOS("tetramerD4h",2.9,0.0,0.0);
      EOS("tetramerTd",2.9,0.0,0.0);
      EOS("pentamerD5h",2.9,0.0,0.0);
      EOS("graphitic",4.1,0.0,0.0);
	*/

      EOS("diam",5.43,0.0,0.0); // diam Si
      EOS("A5",4.370666,4.370666,2.436757); // beta-tin Si
      EOS("bcc",2.829307,0.0,0.0); // bcc Si
      EOS("fcc",3.96095,0.0,0.0); // fcc Si
      EOS("sc",2.8,0.0,0.0); // sc Si
      EOS("wurtzite",3.846203,3.846203,6.366639); // Wurtzite phase
      EOS("A15",4.8,4.8,4.8);
      EOS("dimer",2.4,0.0,0.0);
      EOS("hex",2.4,2.4,0.0);
      EOS("hcp",2.3,0.0,0.0);
      EOS("trimerD3h",2.3,0.0,0.0);
      EOS("trimerC2v",2.3,0.0,0.0);
      EOS("tetramerDih",2.3,0.0,0.0);
      EOS("tetramerD4h",2.3,0.0,0.0);
      EOS("tetramerTd",2.3,0.0,0.0);
      EOS("pentamerD5h",2.3,0.0,0.0);
      EOS("BC8",6.64,0.0,0.0);
      EOS("ST12",6.0,0.0,0.0);
      EOS("cP46",10.27,0.0,0.0);
      EOS("graphitic",3.3,0.0,0.0); 
    } else {
      printf("It seems you are running a FTNN run ...\n");
      printf("Add all layers and rerun to get actual physical properties.\n");
    }


  // Individual timinings.
		//double global_sum_mat_mult;
  //int global_sum_num_mat_mult;
		//MPI_Reduce(&local_sum_mat_mult,&global_sum_mat_mult,1,MPI_DOUBLE,MPI_SUM,0,world);
  //MPI_Reduce(&local_sum_num_mat_mult,&global_sum_num_mat_mult,1,MPI_INT,MPI_SUM,0,world);
		//unsigned long int global_nfunk;
		//MPI_Reduce(&nfunk,&global_nfunk,1,MPI_INT,MPI_SUM,0,world);

    //printf(" Matrix multiplication: %f s\n",global_sum_mat_mult);
    //if (ifga == 0) printf(" Number of energy calls: %ld\n",global_nfunk);
    //else {
    //printf(" Number of energy calls: %ld\n",global_nfunk*nStruc);
    //}

    // Compute end time;
    endtime = MPI_Wtime();
    printf(" EXECUTION TIME ON %d CPU(S): %f s\n",nprocs,(endtime-starttime));
    //fclose(logout);
  }

  // free all dynamically allocated local and global arrays
  freemem();

  MPI_Finalize();

  return 0;
}
