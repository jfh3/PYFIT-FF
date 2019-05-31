#include <stdlib.h>
#include <math.h>
#include "globals.h"
#include "compute.h"
#include "util.h"
#include "NNetInterface.h"
#include "derivatives.h"

void mpi_send_dummy()
{
  int signal = -1;
  for (int i=1; nprocs > 1 && i<nprocs; i++) {
    MPI_Send(&signal,1,MPI_INT,i,1,world);
  }
}


int mpi_exchange_master_nnet(VecDoub &p)
// This routine must be called only by master.
{
  if (me) {
    printf("Error: only master should run this routine .... [%s:%d]\n",
	   __FILE__,__LINE__);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  double *param;

  // Allocate buffer to store parameters in an array.
  param = new double [p.size()];
  for (int i=0; i<p.size(); i++) param[i] = p[i];
  //std::cout << p;
  //std::cout << p.size() << " " << nNNPARAM << "\n";
  //exit(0);

  // Initialize local and global arrays that
  // store sum of partial energies of structures.
  for (int i=0; i<=nStruc; i++) {
    local_eng_sum[i] = 0.0;
    global_eng_sum[i] = 0.0;
  }

  if (PotentialType == 2 && isLocalGrad) {

    // Initialize local and global arrays that store
    // sum of partial derivatives w.r.t. fitting parameters.
    for (int i=0; i<nStruc; i++) {
      for (int j=0; j<nNNPARAM; j++) {
        local_sum_partial_deriv[i*nNNPARAM + j] = 0.0;
        global_sum_partial_deriv[i*nNNPARAM + j] = 0.0;
      }
    }
    // Initialize local and global arrays that store
    // partial derivatives of local BOP parameters w.r.t.
    // fitting parameters involved in constraints.
    for (int i=0; i<nNNPARAM; i++) {
      local_sum_hbconst_partial_deriv[i] = 0.0;
      global_sum_hbconst_partial_deriv[i] = 0.0;
    }
  }

  int signal = isLocalGrad;

  for (int i=0; i<nprocs-1; i++) {
    MPI_Send(&signal,1,MPI_INT,i+1,1,world);
    MPI_Send(&param[0],p.size(),MPI_DOUBLE,i+1,1,world);
  }

  HBconstraint = 0.0;
  VecDoub tmp(MAX_HB_PARAM);
  //double start = MPI_Wtime();
  // Compute energies of atoms.
  switch(PotentialType) {
  case 0: // only BOP
    for (int i=0; i<NAtoms; i++)
      local_eng_sum[atoms[i].gid] += iEnergy(atoms[i].nn,atoms[i].nlist,p);
    break;
  case 1: // only ANN
    // distribute parameters to respective weights and biases
    PartitionNNetParams(p);
    for (int i=0; i<NAtoms; i++)
      local_eng_sum[atoms[i].gid] += NNET_Eng(atoms[i].Gi_list, 1);
    break;
  case 2: // Only PINN
    // distribute parameters to respective weights and biases
    PartitionNNetParams(p);
    for (int i=0; i<NAtoms; i++) {
      // only for analytical derivatives;
      if (isLocalGrad) {
        // Initialize global matrix MatA which
        // stores partial derivatives of Ei w.r.t.
        // local BOP parameters.
        MatA.assign(1, MAX_HB_PARAM, 0.0); // Updated during the energy calculations.
      }
      local_eng_sum[atoms[i].gid] += iNNET_Eng(atoms[i].Gi_list,
                                               atoms[i].nlist,
                                               atoms[i].nn,
                                               tmp);
      // constraint for local BOP parameters.
      HBconstraint += specific_bop_constraint(tmp);
      HBconstraint += mean_sqr(tmp,CONST_HB);

      for (int j=0; j<MAX_HB_PARAM; j++) local_max_pi[i * MAX_HB_PARAM + j] = tmp[j];

      if (isLocalGrad) {
        // Initialize global matrix MatB.
        MatB.assign(MAX_HB_PARAM, nNNPARAM, 0.0);
        // Compute partial derivatives of local BOP parameters
        // w.r.t. to the fitting parameters and store in matrix MatB.
        iCompute_NN_Deriv(PotentialType, MatB);

        // Matrix multiplication: MatA * MatB.
        MatDoub MatC(MatA*MatB);

        for (int k=0; k<nNNPARAM; k++) {
          local_sum_partial_deriv[atoms[i].gid*nNNPARAM + k] += MatC[0][k];
        }

        // Store partial derivatives of local BOP parameters w.r.t.
        // fitting parameters for each atom. This is for HBCONST2.
        for (int j=0; j<MAX_HB_PARAM; j++) {
          for (int k=0; k<nNNPARAM; k++) {
            local_partial_deriv[i*MAX_HB_PARAM*nNNPARAM + j*nNNPARAM + k] = MatB[j][k];
          }
        }
        // ------- Derivatives involved in constraints here ----------
        // This is for specific local BOP parameters.
        if (tmp[big_a] < 0.0)
          for (int k=0; k<nNNPARAM; k++)
            local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[big_a] * MatB[big_a][k];
        if (tmp[alpha] < 0.0)
          for (int k=0; k<nNNPARAM; k++)
            local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[alpha] * MatB[alpha][k];
        if (tmp[big_b] < 0.0)
          for (int k=0; k<nNNPARAM; k++)
            local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[big_b] * MatB[big_b][k];
        if (tmp[beta] < 0.0)
          for (int k=0; k<nNNPARAM; k++)
            local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[beta] * MatB[beta][k];

        // Derivatives related to mean-squared of local BOP parameters.
	if (use_filter == 0) {
	  for (int j=0; j<nNNPARAM; j++) {
	    double sum = 0.0;
	    for (int k=0; k<MAX_HB_PARAM; k++) {
	      sum += 2.0 * tmp[k] * MatB[k][j];
	    }
	    local_sum_hbconst_partial_deriv[j] += CONST_HB * sum / MAX_HB_PARAM; // HBCONST
	    local_sum_hbconst_partial_deriv[j] += CONST_HB2 * sum / MAX_HB_PARAM; // part of HBCONST2
	  }
	}
	if (use_filter == 1) {
	  for (int j=0; j<nNNPARAM; j++) {
	    double sum = 0.0;
	    double sum_2 = 0.0;
	    for (int k=0; k<MAX_HB_PARAM; k++) {
	      sum += 2.0 * (tmp[k] - __BOP_EST[k]) * MatB[k][j];
       sum_2 += 2.0 * (tmp[k] - 2.0 * __BOP_EST[k]) * MatB[k][j];
	    }
	    local_sum_hbconst_partial_deriv[j] += CONST_HB * sum / MAX_HB_PARAM; // HBCONST
     local_sum_hbconst_partial_deriv[j] += CONST_HB2 * sum_2 / MAX_HB_PARAM ; // part of HBCONST2
	  }
	}
        // ------- end of derivatives involved in constraints --------
      }

      // --------- end of calculation of analytical derivatives ------
    }
    break;
  default:
    printf("Error: potential type not supported .... [%s:%d]\n",
	   __FILE__,__LINE__);
    MPI_Abort(world, MPI_ERR_OTHER);
    exit(EXIT_FAILURE);
  }

  //printf(" Rank %d finished energy calculations in %f s.\n",me,MPI_Wtime()-start);
  local_eng_sum[nStruc] = HBconstraint;

  // Sum energies of atoms within each group.
  MPI_Reduce(local_eng_sum,global_eng_sum,nStruc+1,MPI_DOUBLE,MPI_SUM,0,world);

  // Update energies of structures.
  for (int i=0; i<nStruc; i++) Lattice[i].E = global_eng_sum[i];
  HBconstraint = global_eng_sum[nStruc];

  if (PotentialType == 2) {
    int size = max_num_atoms_per_proc * MAX_HB_PARAM;
    MPI_Gather(local_max_pi, size, MPI_DOUBLE,
	       global_max_pi, size, MPI_DOUBLE, 0, world);

    // Only for debugging purpose.
    /*for (int i=0; i<nprocs; i++) { // loop over blocks.
      printf(" Rank %d: %f %f %f %f\n",i,
	     global_max_pi[i*max_num_atoms_per_proc*MAX_HB_PARAM+0],
	  global_max_pi[i*max_num_atoms_per_proc*MAX_HB_PARAM+1],
	  global_max_pi[i*max_num_atoms_per_proc*MAX_HB_PARAM+2],
	  global_max_pi[i*max_num_atoms_per_proc*MAX_HB_PARAM+3]);
    }
    exit(0);*/
    if (isLocalGrad) {

      // Sum partial derivatives w.r.t. fitting parameters within each group.
      MPI_Reduce(local_sum_partial_deriv,global_sum_partial_deriv,
		 nStruc*nNNPARAM,MPI_DOUBLE,MPI_SUM,0,world);

      // Sum partial derivatives w.r.t. fitting parameters involved in HBCONST1.
      MPI_Reduce(local_sum_hbconst_partial_deriv,global_sum_hbconst_partial_deriv,
		 nNNPARAM,MPI_DOUBLE,MPI_SUM,0,world);
      // Collect all derivatives of local BOP w.r.t. fitting parameters
      // of all atoms. This is for HBCONST2.
      size = max_num_atoms_per_proc * MAX_HB_PARAM * nNNPARAM;
      MPI_Gather(local_partial_deriv, size, MPI_DOUBLE,
		 global_partial_deriv, size, MPI_DOUBLE, 0, world);
    }
  }

  delete [] param;

  return 0;
}


int mpi_exchange_slaves_nnet()
// This routine must be called only by slaves.
{
  if (me == 0) {
    printf("Error: only slaves should run this routine from file %s at line %d in function %s\n",
	   __FILE__,__LINE__,__FUNCTION__);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }

  MPI_Status status;
  double *param;
  int npar;
  int signal;

  // Allocate buffer to store fitting parameters.
  if (PotentialType == 0) npar = MAX_BOP_PARAM;
  if (PotentialType == 1 || PotentialType == 2) npar = nNNPARAM;
  param = new double [npar];

  status.MPI_TAG = 1; // first time.

  while (status.MPI_TAG) {
    // termination signal.
    MPI_Recv(&signal,1,MPI_INT,0,1,world,&status);
    if (signal == -1) {
      status.MPI_TAG = 0;
      continue;
    }

    // If signal is zero, just compute energies.
    // If it is 1, compute energies and analytical derivatives.
    isLocalGrad = signal;

    // Initialize the constraint to local BOP parameters.
    HBconstraint = 0.0;

    // Initialize local array that stores sum
    // of partial energies of structures.
    for (int i=0; i<=nStruc; i++) local_eng_sum[i] = 0.0;

    if (PotentialType == 2 && isLocalGrad) {

      // Initialize local array that stores sum of
      // partial derivatives w.r.t. fitting parameters.
      for (int i=0; i<nStruc; i++) {
	for (int j=0; j<nNNPARAM; j++) {
	  local_sum_partial_deriv[i*nNNPARAM + j] = 0.0;
	}
      }

      // Initialize local array that stores
      // partial derivatives of local BOP parameters w.r.t.
      // fitting parameters involved in constraints.
      for (int i=0; i<nNNPARAM; i++) {
	local_sum_hbconst_partial_deriv[i] = 0.0;
      }

    }

    // Receive parameters from the Master.
    MPI_Recv(&param[0],npar,MPI_DOUBLE,0,1,world,&status);

    // Update parameter vector.
    ParamVec.assign(npar,param);
    VecDoub tmp(MAX_HB_PARAM);
    //double start = MPI_Wtime();
    //printf(" Rank %d computing energies ...\n",me);
    // Compute energies of atoms.
    switch(PotentialType) {
    case 0: // only BOP
      for (int i=0; i<NAtoms; i++) {
	local_eng_sum[atoms[i].gid] += iEnergy(atoms[i].nn, atoms[i].nlist, ParamVec);
      }
      break;
    case 1: // only NN
      // distribute parameters to respective weights and biases
      PartitionNNetParams(ParamVec);
      for (int i=0; i<NAtoms; i++)
	local_eng_sum[atoms[i].gid] += NNET_Eng(atoms[i].Gi_list,1);
      break;
    case 2: // only PINN
      // Distribute parameters to respective weights and biases
      PartitionNNetParams(ParamVec);

      for (int i=0; i<NAtoms; i++) {
	// only for analytical derivatives;
	if (isLocalGrad) {
	  // Initialize global matrix MatA which
	  // stores partial derivatives of Ei w.r.t.
	  // local BOP parameters.
	  MatA.assign(1,MAX_HB_PARAM,0.0);
	}

	local_eng_sum[atoms[i].gid] += iNNET_Eng(atoms[i].Gi_list,
						 atoms[i].nlist,
						 atoms[i].nn,tmp);

	// constraint for local BOP parameters.
	HBconstraint += specific_bop_constraint(tmp);
	HBconstraint += mean_sqr(tmp,CONST_HB);

	for (int j=0; j<MAX_HB_PARAM; j++) local_max_pi[i*MAX_HB_PARAM + j] = tmp[j];

	if (isLocalGrad) {
	  // Initialize global matrix MatB.
	  MatB.assign(MAX_HB_PARAM,nNNPARAM,0.0);
	  // Compute partial derivatives of local BOP parameters
	  // w.r.t. to the fitting parameters and store in matrix MatB.
	  iCompute_NN_Deriv(PotentialType,MatB);
	  // Matrix multiplication: MatA * MatB.
	  MatDoub MatC(MatA*MatB);
	  for (int k=0; k<nNNPARAM; k++) {
	    local_sum_partial_deriv[atoms[i].gid*nNNPARAM + k] += MatC[0][k];
	  }
	  // Store partial derivatives of local BOP parameters w.r.t.
	  // fitting parameters for each atom. This is for HBCONST2.
	  for (int j=0; j<MAX_HB_PARAM; j++) {
	    for (int k=0; k<nNNPARAM; k++) {
	      local_partial_deriv[i*MAX_HB_PARAM*nNNPARAM + j*nNNPARAM + k] = MatB[j][k];
	    }
	  }
	  // ------- Derivatives involved in constraints here ----------
	  // This is for specific local BOP parameters.
	  if (tmp[big_a] < 0.0)
	    for (int k=0; k<nNNPARAM; k++) local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[big_a] * MatB[big_a][k];
	  if (tmp[alpha] < 0.0)
	    for (int k=0; k<nNNPARAM; k++) local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[alpha] * MatB[alpha][k];
	  if (tmp[big_b] < 0.0)
	    for (int k=0; k<nNNPARAM; k++) local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[big_b] * MatB[big_b][k];
	  if (tmp[beta] < 0.0)
	    for (int k=0; k<nNNPARAM; k++) local_sum_hbconst_partial_deriv[k] += 8.0 * tmp[beta] * MatB[beta][k];

	  /*for (int k=0; k<nNNPARAM; k++) {
	    local_sum_hbconst_partial_deriv[k] += -2.0 * pow(tmp[big_a]-fabs(tmp[big_a]),2) / fabs(fabs(tmp[big_a])) * MatB[big_a][k];
	    local_sum_hbconst_partial_deriv[k] += -2.0 * pow(tmp[alpha]-fabs(tmp[alpha]),2) / fabs(fabs(tmp[alpha])) * MatB[alpha][k];
	    local_sum_hbconst_partial_deriv[k] += -2.0 * pow(tmp[big_b]-fabs(tmp[big_b]),2) / fabs(fabs(tmp[big_b])) * MatB[big_b][k];
	    local_sum_hbconst_partial_deriv[k] += -2.0 * pow(tmp[beta]-fabs(tmp[beta]),2) / fabs(fabs(tmp[beta])) * MatB[beta][k];
	  }*/
	  // Mean squared of local BOP parameters.
	  for (int j=0; j<nNNPARAM; j++) {
	    double sum{0.0};
	    for (int k=0; k<MAX_HB_PARAM; k++) {
	      sum += 2.0 * tmp[k] * MatB[k][j];
	    }
	    local_sum_hbconst_partial_deriv[j] += CONST_HB * sum / MAX_HB_PARAM; // HBCONST
					local_sum_hbconst_partial_deriv[j] += CONST_HB2 * sum / MAX_HB_PARAM; // part of HBCONST2.
	  }
	  // ------- end of derivatives involved in constraints --------
	}
	// ------ end of calculation of analytical derivatives --------
      }
      break;
    default:
      printf("Error: this potential type is not currently supported from file %s at line %d in function %s\n",
	     __FILE__,__LINE__,__FUNCTION__);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    //printf(" Rank %d finished energy calculations in %f s.\n",me,MPI_Wtime()-start);

    local_eng_sum[nStruc] = HBconstraint;
    // Sum energies of atoms within each group.
    MPI_Reduce(local_eng_sum,global_eng_sum,nStruc+1,
	       MPI_DOUBLE,MPI_SUM,0,world);
    if (PotentialType == 2) {
      int size = max_num_atoms_per_proc*MAX_HB_PARAM;
      MPI_Gather(local_max_pi,size,MPI_DOUBLE,
		 global_max_pi,size,MPI_DOUBLE,0,world);

      // Only for debugging purpose.
      //printf(" Rank %d: %f %f %f %f\n",me,
      //local_max_pi[0],local_max_pi[1],
      //local_max_pi[2],local_max_pi[3]);

      if (isLocalGrad) {

	// Sum partial derivatives w.r.t. fitting parameters within each group.
	MPI_Reduce(local_sum_partial_deriv,global_sum_partial_deriv,
		   nStruc*nNNPARAM,MPI_DOUBLE,MPI_SUM,0,world);

	// Sum partial derivatives w.r.t. fitting parameters involved in HBCONST1.
	MPI_Reduce(local_sum_hbconst_partial_deriv,global_sum_hbconst_partial_deriv,
		   nNNPARAM,MPI_DOUBLE,MPI_SUM,0,world);

	// Send derivatives of local BOP w.r.t. fitting parameters
	// of atoms in this process for collection at the root.
	// This is for HBCONST2.
	size = max_num_atoms_per_proc*MAX_HB_PARAM*nNNPARAM;
	MPI_Gather(local_partial_deriv,size,MPI_DOUBLE,
		   global_partial_deriv,size,MPI_DOUBLE,0,world);
      }
    }
    //printf("I am rank %d from file %s at line %d ....\n",me,__FILE__,__LINE__);
  }

  delete [] param;

  return 0; // Not used.
}


/*void mpi_dist_data()
{
  // -------- Distribute training data to cpus --------
  if (me == 0) {
    int numsent;
    int blk_shft;
    int Gi_shft;
    int i, l;
    double *packet;
    int packet_size;
    int gi_size;
    int ncount;

    if (noGi == 1) gi_size = layer[0].nnodes;
    else gi_size = nSigmas*MAX_LSP;

    packet = NULL;
    blk_shft = 0;
    Gi_shft = 0;
    l = 0; // runs over number of bases of each configuration.
    i = 0; // runs over number of configurations.
    numsent = 0;
    ncount = 0;
    NAtoms = Ntotal_bases/nprocs;

    atoms = create(atoms,NAtoms,"mpi_dist_data():create");

    // Master keeps first NAtoms atoms.
    while (ncount < NAtoms) {
      if (l) {
	blk_shft += 4*Lattice[i].nneighbors[l-1];
	Gi_shft += gi_size;
      }
      atoms[ncount].gid = i; // group ID.
      atoms[ncount].nn = Lattice[i].nneighbors[l]; // no. of neighbors.
      atoms[ncount].Gi_list = create(atoms[ncount].Gi_list,gi_size,"mpi_dist_data():create");
      for (int n=0; n<gi_size; n++) atoms[ncount].Gi_list[n] = Lattice[i].G_local[Gi_shft+n]; // Gis
      atoms[ncount].nlist = create(atoms[ncount].nlist,4*atoms[ncount].nn,"mpi_dist_data():create");
      for (int n=0; n<4*Lattice[i].nneighbors[l]; n++)
	atoms[ncount].nlist[n] = Lattice[i].neighbors[blk_shft+n]; // neighborlist.
      l++;
      ncount++;
      if (l >= Lattice[i].nbas) {
	l = 0;
	blk_shft = 0;
	Gi_shft = 0;
	i++;
      }
    }

    //printf(" l=%d i=%d blk_shft=%d Gi_shft=%d\n",l,i,blk_shft,Gi_shft);
    //printf("Total number of bases = %d .... in file %s at line %d\n",Ntotal_bases,__FILE__,__LINE__);

    while (ncount < Ntotal_bases) {
      //for (int proc=0; proc<nprocs-1 && ncount < Ntotal_bases; proc++) {
      for (int proc=0; proc<nprocs-1 && ncount < Ntotal_bases; proc++) {
	if (l) {
	  blk_shft += 4*Lattice[i].nneighbors[l-1];
	  Gi_shft += gi_size;
	}
	// message packet format for each atom.
	// [configuration id][no. of neighbors][Gi list][neighbor list]
	// packet size: [1][1][nSigmas*MAX_LSP][4*[packet[1]]
	packet_size = 2 + gi_size + 4*Lattice[i].nneighbors[l];
	// Allocate memory for packet to send.
	packet = grow(packet,packet_size,"mpi_exchange_master_nnet():grow");
	// Pack all relavent data.
	packet[0] = (double) i;
	packet[1] = (double) Lattice[i].nneighbors[l]; // no. of neighbors.
	for (int n=0; n<gi_size; n++) packet[n+2] = Lattice[i].G_local[Gi_shft+n]; // Gis
	for (int n=0; n<4*Lattice[i].nneighbors[l]; n++)
	  packet[n+2+gi_size] = Lattice[i].neighbors[blk_shft+n]; // neighborlist.

	// Send whole packet.
	MPI_Send(packet,packet_size,MPI_DOUBLE,proc+1,1,world);

	numsent++;
	ncount++;
	l++;
	if (l >= Lattice[i].nbas) {
	  l = 0;
	  blk_shft = 0;
	  Gi_shft = 0;
	  i++;
	}
	//printf(" l=%d i=%d blk_shft=%d Gi_shft=%d\n",l,i,blk_shft,Gi_shft);
      }
    }

    // Only for debugging purpose.
#ifdef DEBUG
    char tmpstr[1024];
    sprintf(tmpstr,"Rank %d kept %d out of %d and sent %d atoms to slaves.\n",
	    me, NAtoms, Ntotal_bases, numsent);
    printf(tmpstr);
    fflush(stdout);
#endif
    // Finally send MPI_TAG = 0 to signal slaves to close communication.
    for (int proc=1; proc<nprocs; proc++) {
      MPI_Send(packet,packet_size,MPI_DOUBLE,proc,0,world);
    }

    //MPI_Allreduce(NAtoms,max_num_atoms_per_proc,1,MPI_INT,MPI_MAX,0,world);
    //MPI_Gather(&NAtoms,1,MPI_INT,num_atoms_per_proc,1,MPI_INT,0,world);

    sfree(packet);

  } else {
    // Slave part.
    double *packet;
    int numrecv;
    MPI_Status status;
    int gi_size; // = nSigmas*MAX_LSP;
    int packet_size;
    int nb;
    int count;

    if (noGi == 1) gi_size= layer[0].nnodes;
    else gi_size = nSigmas*MAX_LSP;

    packet = NULL;
    numrecv = 0;
    status.MPI_TAG = 1;

    atoms = create(atoms,numrecv+1,"mpi_dist_data():create");

    while (status.MPI_TAG) {
      // Probe for an incoming message from process zero.
      MPI_Probe(0,MPI_ANY_TAG,world,&status);

      // When probe returns, the status object has the size and
      // other attributes of the incoming message. Get the message count.

      MPI_Get_count(&status,MPI_DOUBLE,&packet_size);
      if (status.MPI_TAG == 0) continue;
      if (packet_size == 0) {
	printf("Error: too few messages received from file %s at line %d\n",__FILE__,__LINE__);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }
      //printf(" Rank %d received %d of double\n",me,packet_size);
      // Allocate a buffer to hold the incoming message.
      packet = grow(packet,packet_size,"mpi_exchange_slaves_nnet():grow");

      // Now receive the message in the allocated buffer.
      MPI_Recv(packet,packet_size,MPI_DOUBLE,0,MPI_ANY_TAG,world,&status);

      // Get actual message count.
      MPI_Get_count(&status,MPI_DOUBLE,&count);
      // Some checks.
      if (count < packet_size) {
	errmsg("received different amount of message",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }

      nb = (int) packet[1];
      //printf("%d %d %d.\n",me,nb,(int)packet[0]);
      //std::cout << " number of bases = " << nb << "\n";
      count = 2 + gi_size + 4*nb;
      if (count != packet_size) {
	errmsg("did not receive correct amount of message",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }

      // Allocate buffer to hold neighborlist do some checks.
      if (nb) {
	atoms[numrecv].nlist = create(atoms[numrecv].nlist,4*nb,"mpi_dist_data():create");
      } else {
	errmsg("zero neighborlist in the message",FERR);
	if (nprocs > 1) MPI_Abort(world,errcode);
	exit(EXIT_FAILURE);
      }

      // Allocate buffer to hold Gis.
      atoms[numrecv].Gi_list = create(atoms[numrecv].Gi_list,gi_size,"mpi_dist_data():grow");

      // Now unpack the message.
      atoms[numrecv].gid = (int) packet[0];
      atoms[numrecv].nn = nb;
      for (int i=0; i<gi_size; i++) atoms[numrecv].Gi_list[i] = packet[2+i];
      for (int i=0; i<4*nb; i++) atoms[numrecv].nlist[i] = packet[2+gi_size+i];

      numrecv++;
      atoms = grow(atoms,numrecv+1,"mpi_dist_data():create");
    }

    atoms = grow(atoms,numrecv,"mpi_dist_data():create");
    NAtoms = numrecv;

    // Only for debugging purpose.
#ifdef DEBUG
    char tmpstr[1024];
    sprintf(tmpstr,"Rank %d received %d atoms ...\n",me,NAtoms);
    printf(tmpstr);
    fflush(stdout);
#endif
    //MPI_Allreduce(NAtoms,max_num_atoms_per_proc,1,MPI_INT,MPI_MAX,0,world);
    //MPI_Gather(&NAtoms,1,MPI_INT,num_atoms_per_proc,1,MPI_INT,0,world);

    sfree(packet);
  }
}
*/

void mpi_dist_data3()
{
  // -------- Distribute training data to cpus --------

  double *packet_double;
  int packet_size;
  int nInt_size;
  int nDoub_size;
  int ncount;
  MPI_Status status;

  if (me == 0) {
    printf("Distributing atoms to %d cpus ....\n", nprocs);
    int numsent;
    int blk_shft;
    int i, l;

    packet_double = NULL;
    blk_shft = 0;
    l = 0; // runs over number of bases of each configuration.
    i = 0; // runs over number of configurations.
    numsent = 0;
    ncount = 0;
    NAtoms = Ntotal_bases/nprocs;

    atoms = create(atoms, NAtoms, "mpi_dist_data():create");

    // root keeps first NAtoms atoms.
    while (ncount < NAtoms) {
      if (l) {
	blk_shft += 4*Lattice[i].nneighbors[l-1];
      }
      atoms[ncount].atomid = ncount;
      atoms[ncount].gid = i;                          // group/configuration ID.
      atoms[ncount].nn = Lattice[i].nneighbors[l];    // no. of neighbors.

      atoms[ncount].nlist = create(atoms[ncount].nlist,
				   4 * atoms[ncount].nn,
				   "mpi_dist_data():create");
      for (int n=0; n<4*Lattice[i].nneighbors[l]; n++)
	atoms[ncount].nlist[n] = Lattice[i].neighbors[blk_shft+n]; // neighborlist.
      atoms[ncount].Gi_list = NULL;
      l++;
      ncount++;
      if (l >= Lattice[i].nbas) {
	l = 0;
	blk_shft = 0;
	i++;
      }
    }

    //printf("Rank %d received %d atoms ....\n", me, NAtoms);
    //printf(" l=%d i=%d blk_shft=%d Gi_shft=%d\n",l,i,blk_shft,Gi_shft);
    //printf("Total number of bases = %d .... in file %s at line %d\n",Ntotal_bases,__FILE__,__LINE__);

    while (ncount < Ntotal_bases) {
      for (int proc=0; proc<nprocs-1 && ncount < Ntotal_bases; proc++) {
	if (l) {
	  blk_shft += 4*Lattice[i].nneighbors[l-1];
	}

	// Pack all relavent data of each atom into one packet.
	// [atom id] [configuration id] [# of neighbors] [neighbor list]
	// packet size: [1] + [1] + [1] + [4 * nneighbors]
	nInt_size = 3;
	nDoub_size = 4 * Lattice[i].nneighbors[l];
	packet_size = nInt_size + nDoub_size;
	// Allocate memory for the packet.
	if (packet_double) packet_double = grow(packet_double, packet_size, "");
	else packet_double = create(packet_double, packet_size, "");
	// Pack data.
	packet_double[0] = (double) ncount;
	packet_double[1] = (double) i;                         // group/configuration ID.
	packet_double[2] = (double) Lattice[i].nneighbors[l];  // # of neighbors.

	for (int n=0; n<nDoub_size; n++)
	  packet_double[nInt_size + n] = Lattice[i].neighbors[blk_shft + n];

	// Send the packet.
	MPI_Send(packet_double, packet_size, MPI_DOUBLE, proc+1, 1, world);
	//printf("Rank %d sent %d doubles to rank %d .... [%d %d]\n", me, packet_size, proc+1, i, l);


	numsent++;
	ncount++;
	l++;
	if (l >= Lattice[i].nbas) {
	  l = 0;
	  blk_shft = 0;
	  i++;
	}
	//printf(" l=%d i=%d blk_shft=%d Gi_shft=%d\n",l,i,blk_shft,Gi_shft);
      }
    }

    // Finally send MPI_TAG = 0 to signal slaves to close communication.
    for (int proc=1; proc<nprocs; proc++) {
      MPI_Send(&NAtoms, 1, MPI_INT, proc, 0, world);
    }

    //MPI_Allreduce(NAtoms,max_num_atoms_per_proc,1,MPI_INT,MPI_MAX,0,world);
    //MPI_Gather(&NAtoms,1,MPI_INT,num_atoms_per_proc,1,MPI_INT,0,world);

  } else { // ------- Slave part stars here --------

    int numrecv;
    int nb;
    //char errstring[128];
    //int errlen;

    packet_double = NULL;
    numrecv = 0;
    status.MPI_TAG = 1;

    atoms = create(atoms, numrecv+1, "mpi_dist_data2():create");

    while (status.MPI_TAG) {
      // Probe for an incoming message from process zero.
      MPI_Probe(0, MPI_ANY_TAG, world, &status);

      if (status.MPI_TAG == 0) continue;

      // When probe returns, the status object has the size and
      // other attributes of the incoming message.
      // Get the message count.
      MPI_Get_count(&status, MPI_DOUBLE, &packet_size);

      if (packet_size == 0) {
	printf("Error: too few messages received from file %s at line %d\n",
	       __FILE__, __LINE__);
	MPI_Abort(world, MPI_ERR_OTHER);
	exit(EXIT_FAILURE);
      }
      //printf(" Rank %d received %d of double\n",me,packet_size);

      // Allocate a buffer to hold int packet.
      if (packet_double) packet_double = grow(packet_double, packet_size, "");
      else packet_double = create(packet_double, packet_size, "");

      // Now receive the message in the allocated buffer.
      MPI_Recv(packet_double, packet_size, MPI_DOUBLE, 0 , 1, world, &status);
      //MPI_Error_string(status.MPI_ERROR, errstring, &errlen);
      //printf("slave %d: [%d:%s:%d]\n", me, status.MPI_ERROR, errstring, errlen);
      //printf("slave %d received %d doubles from rank %d ....\n", me, packet_size, status.MPI_SOURCE);

      // Get actual message count.
      MPI_Get_count(&status, MPI_DOUBLE, &ncount);
      // Some checks.
      if (ncount < packet_size) {
	errmsg("received different amount of message",FERR);
	MPI_Abort(world, errcode);
	exit(EXIT_FAILURE);
      }

      // Now unpack first packet.
      atoms[numrecv].atomid = (int) packet_double[0];
      atoms[numrecv].gid = (int) packet_double[1];
      atoms[numrecv].nn = nb = (int) packet_double[2];
      nInt_size = 3;
      nDoub_size = 4 * nb;
      // Allocate buffers to hold neighbor types and neighbor list.
      if (nb) {
	atoms[numrecv].nlist = create(atoms[numrecv].nlist, nDoub_size, "mpi_dist_data2():create");
      } else {
	printf("Error: invalid number of neighbors .... [%s:%d]\n",
	       __FILE__, __LINE__);
	MPI_Abort(world, errcode);
	exit(EXIT_FAILURE);
      }
      atoms[numrecv].Gi_list = NULL;
      for (int i=0; i<nDoub_size; i++) atoms[numrecv].nlist[i] = packet_double[nInt_size + i];

      numrecv++;
      atoms = grow(atoms, numrecv+1, "mpi_dist_data2():create");
    }

    atoms = grow(atoms, numrecv, "mpi_dist_data2():create");
    NAtoms = numrecv;

    // Only for debugging purpose.
    //printf("Rank %d received %d atoms ....\n", me, NAtoms);

    //MPI_Allreduce(NAtoms,max_num_atoms_per_proc,1,MPI_INT,MPI_MAX,0,world);
    //MPI_Gather(&NAtoms,1,MPI_INT,num_atoms_per_proc,1,MPI_INT,0,world);
  }

  if (packet_double) sfree(packet_double);
}
