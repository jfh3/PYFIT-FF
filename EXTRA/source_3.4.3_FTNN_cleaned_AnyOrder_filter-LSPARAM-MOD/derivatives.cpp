#include "globals.h"
#include "NNetInterface.h"
#include "derivatives.h"
#include "util.h"

void compute_SMat()
{
  // Initialize SMat array.
  Init_SMat();

  int indx_last_hidden = nLayers - 2;
  int indx_output = nLayers - 1;

  for (int l=indx_last_hidden; l>0; l--) {
    if (l == indx_last_hidden) {
      for (int i=0; i<layer[l].nnodes; i++) { // rows
	for (int j=0; j<layer[indx_output].nnodes; j++) { // cols
	  layer[l].SMat[i*layer[indx_output].nnodes + j] = layer[l].fdot[i] *
	      layer[l+1].Weights[i*layer[indx_output].nnodes + j];
	}
      }
      //WriteMatrix(logout,layer[l].s,layer[l-1].nnodes,layer[l].nnodes);
    } else {
      for (int i=0; i<layer[l].nnodes; i++) { // rows
	for (int j=0; j<layer[indx_output].nnodes; j++) { // cols
	  layer[l].SMat[i*layer[indx_output].nnodes + j] = 0.0;
	  for (int k=0; k<layer[l+1].nnodes; k++) {
	    layer[l].SMat[i*layer[indx_output].nnodes + j] +=
		layer[l+1].Weights[i*layer[l+1].nnodes + k] * layer[l+1].SMat[k*layer[indx_output].nnodes + j];
	  }
	  layer[l].SMat[i*layer[indx_output].nnodes + j] *= layer[l].fdot[i];
	}
      }
      //WriteMatrix(logout,layer[l].s,layer[l-1].nnodes,layer[l].nnodes);
    }
  }
}

void computeNNDeriv(VecDoub &x, VecDoub &g)
{
  int size = layer[0].nnodes;

  if (PotentialType != 1) {
    printf("Error: potential type (%d) not supported .... [%s:%d]\n",
	   PotentialType, __FILE__, __LINE__);
    MPI_Abort(world, MPI_ERR_OTHER);
    exit(EXIT_FAILURE);
  }

  // Initialize the gradient vector.
  for (int i=0; i<g.size(); i++) g[i] = 0.0;

  PartitionNNetParams(x);

  MatDoub tmp_grad;
  int offset = 0;

  for (int s=0; s<nStruc; s++) {
    tmp_grad.assign(layer[nLayers-1].nnodes, g.size(), 0.0);
    if (s) offset += Lattice[s - 1].nbas;
    for (int i=0; i<Lattice[s].nbas; i++) {
      // Update vsum of layer[0]
      for (int j=0; j<size; j++) layer[0].vsum[j] = __GiList[offset + i].gilist[j];
      iCompute_NN_Deriv(PotentialType, tmp_grad);
    }
    for (int k=0; k<layer[nLayers-1].nnodes; k++) {
      for (int n=0; n<g.size(); n++) {
	g[n] += tmp_grad[k][n] * Lattice[s].w * (Lattice[s].E - Lattice[s].E0)/Lattice[s].nbas/Lattice[s].nbas;
      }
    }
  }

  /*for (int s=0; s<nStruc; s++) {
    tmp_grad.assign(layer[nLayers-1].nnodes, g.size(), 0.0);
    for (int i=0; i<Lattice[s].nbas; i++) {
      // Update vsum of layer[0]
      for (int j=0; j<size; j++) layer[0].vsum[j] = Lattice[s].G_local[i*size + j];
      iCompute_NN_Deriv(PotentialType, tmp_grad);
    }
    for (int k=0; k<layer[nLayers-1].nnodes; k++) {
      for (int n=0; n<g.size(); n++) {
	g[n] += tmp_grad[k][n] * Lattice[s].w * (Lattice[s].E - Lattice[s].E0)/Lattice[s].nbas/Lattice[s].nbas;
      }
    }
  }*/

  // Add partial derivatives from NN constraints.
  for (int n=0; n<g.size(); n++) {
    g[n] = 2.0 * g[n] / nStruc + 2.0 * CONST_NN * x[n] / nNNPARAM;
  }
}

void iCompute_NN_Deriv(const int pottype, MatDoub &g)
{
  //
  // Make sure inputs are updated before calling this subroutine !!!
  //

  // Compute derivatives of logistic function at each nodes of each layer.
  evaluate_nnet();

  // Compute S matrices.
  compute_SMat();

  for (int i=0; i<layer[nLayers-1].nnodes; i++) { // loop over no. of output nodes
    int j = 0;
    // loop over layers of neural network.
    // The first layer is the input layer and skipped.
    for (int l=1; l<nLayers; l++) {
      // gradient matrix corresponding to weights: m x n matrix
      for (int m=0; m<layer[l-1].nnodes; m++) { // rows
	for (int n=0; n<layer[l].nnodes; n++) { // cols
	  if (l == nLayers-1) {
	    if (i == n) g[i][j] += layer[l-1].vsum[m];
	  } else {
	    g[i][j] += layer[l-1].vsum[m] * layer[l].SMat[n*layer[nLayers-1].nnodes+i];
	  }
	  j++;
	}
      }
      // biases: vector of size nnodes.
      for (int m=0; m<layer[l].nnodes; m++) { // cols
	if (l == nLayers-1) {
	  if (i == m) g[i][j] += 1.0;
	} else {
	  g[i][j] += layer[l].SMat[m*layer[nLayers-1].nnodes+i];
	}
	j++;
      }
    }
    if (j != nNNPARAM) {
      printf("Error: condition not satisfied ... [%s:%d]\n",
	     __FILE__, __LINE__);
      MPI_Abort(world, MPI_ERR_OTHER);
      exit(EXIT_FAILURE);
    }
  }
}

void computePINNDeriv(VecDoub &x, VecDoub &g)
{
  // Initialize the gradient vector.
  for (int i=0; i<g.size(); i++) g[i] = 0.0;

  for (int i=0; i<nStruc; i++) {
    for (int j=0; j<g.size(); j++) {
      g[j] += global_sum_partial_deriv[i*nNNPARAM + j] * Lattice[i].w *
	  (Lattice[i].E - Lattice[i].E0)/Lattice[i].nbas/Lattice[i].nbas;
    }
  }

  // Add partial derivatives due to constraints NN and HB.
  for (int i=0; i<g.size(); i++) {
    g[i] = 2.0 * g[i] / nStruc + 2.0 * CONST_NN * x[i] / nNNPARAM +
	global_sum_hbconst_partial_deriv[i] / Ntotal_bases;
  }

  // ------------- Calculations related to constraint HB2 --------------------
  // Vector to store average of each local BOP over all atoms.
  VecDoub tmp_g(g.size(),0.0);
  if (use_filter == 0) {
    VecDoub ave_p(MAX_HB_PARAM, 0.0);
    // Sum each local BOP over all atoms.
    for (int i=0; i<nprocs; i++) {
      for (int j=0; j<num_atoms_per_proc[i]; j++) {
	for (int k=0; k<MAX_HB_PARAM; k++) {
	  int n = i*max_num_atoms_per_proc*MAX_HB_PARAM + j*MAX_HB_PARAM + k;
	  ave_p[k] += global_max_pi[n];
	}
      }
    }

    // Compute averages.
    for (int k=0; k<MAX_HB_PARAM; k++) {
      ave_p[k] /= Ntotal_bases;
    }

    // Matrix to store partial derivatives of averaged local BOP w.r.t. fitting parameters.
    MatDoub dave_p(MAX_HB_PARAM, nNNPARAM, 0.0);
    // Compute averages.
    for (int k=0; k<MAX_HB_PARAM; k++) {
      for (int l=0; l<nNNPARAM; l++) {
	for (int i=0; i<nprocs; i++) {
	  for (int j=0; j<num_atoms_per_proc[i]; j++) {
	    int n = i*max_num_atoms_per_proc*MAX_HB_PARAM*nNNPARAM +
		j*MAX_HB_PARAM*nNNPARAM + k*nNNPARAM + l;
	    dave_p[k][l] += global_partial_deriv[n];
	  }
	}
	dave_p[k][l] /= Ntotal_bases;
      }
    }

    for (int j=0; j<nNNPARAM; j++) {
      for (int i=0; i<MAX_HB_PARAM; i++) {
	tmp_g[j] += -2.0 * ave_p[i] * dave_p[i][j];
      }
      tmp_g[j] *= CONST_HB2 / MAX_HB_PARAM;
    }
  }

  /*for (int l=0; l<nNNPARAM; l++) {
    for (int i=0; i<nprocs; i++) {
      for (int j=0; j<num_atoms_per_proc[i]; j++) {
	for (int k=0; k<MAX_HB_PARAM; k++) {
	  int m = i*max_num_atoms_per_proc*MAX_HB_PARAM*nNNPARAM +
	      j*MAX_HB_PARAM*nNNPARAM + k*nNNPARAM + l;
	  int n = i*max_num_atoms_per_proc*MAX_HB_PARAM + j*MAX_HB_PARAM + k;
	  tmp_g[l] += (global_max_pi[n] - ave_p[k]) * (global_partial_deriv[m] - dave_p[k][l]);
	}
      }
    }
    tmp_g[l] *= 2.0 * CONST_HB2 / (Ntotal_bases * MAX_HB_PARAM);
  }*/

  // Add contribution from the second constraint related to local BOP parameters.
  for (int i=0; i<g.size(); i++) {
    g[i] += tmp_g[i];
  }
}

void Init_SMat()
{
  int indx_last = nLayers - 1; // index of output layer.
  for (int i=1; i<nLayers; i++) {
    if (i != indx_last) {
      int size = layer[i].nnodes*layer[indx_last].nnodes;
      for (int j=0; j<size; j++) layer[i].SMat[j] = 0.0;
    }
  }
}
