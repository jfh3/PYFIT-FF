#ifndef MPI_STUFF_H
#define MPI_STUFF_H

//#include "compute.h"
//#include "globals.h"
#include "util.h"

int mpi_exchange_master (VecDoub &p);
int mpi_exchange_master_nnet (VecDoub &p);
int mpi_exchange_slaves ();
int mpi_exchange_slaves_nnet ();
void mpi_send_dummy();
//void mpi_dist_data();
void mpi_dist_data3();

#endif // MPI_STUFF_H
