from sys import path
path.append("../subroutines")

from   Config                 import *
from   PoscarLoader           import PoscarDataFile
from   NeighborList           import generateNeighborList
from   TrainingSet            import TrainingSetFile, WriteTrainingSet
from   NeuralNetwork          import NeuralNetwork
from   PyTorchNet             import TorchNet
from   ConfigurationParser    import TrainingFileConfig
from   StructuralParameters   import GenerateStructuralParameters
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import matplotlib.pyplot   as plt
import numpy               as np
import Util
import sys
import os
import copy
from   Util import log, log_indent, log_unindent, ProgressBar
from   Help import help_str
from   time import time

def TrainNetwork(force_cpu, randomize_nn):

	# The primary steps for loading are as follows:
	#     1) Load the neural network weights and biases.
	#     2) Load the LSParam file that contains Gi's and
	#        DFT energies for each structure.

	neural_network_data = NeuralNetwork(NEURAL_NETWORK_FILE) 
	training_set        = TrainingSetFile(TRAINING_SET_FILE)

	
	
	
	# Randomly select a set of structure IDs to use as the training set
	# and use the rest as a validation set.
	all_indices          = np.array(range(training_set.n_structures))
	all_indices          = list(all_indices)
	training_indices     = list(all_indices)
	n_training_indices   = len(training_indices)
	

	# We need to know how many atoms are part of the training set and 
	# how many are part of the validation set.
	n_train_atoms = 0
	for index in training_indices:
		n_train_atoms += len(training_set.training_structures[index])

	# The following code assumes that the values in the LSParam file are
	# ordered sequentially, first by structure, then by atom.

	

	# This reduction matrix will be multiplied by the output column vector
	# to reduce the energy of each atom to the energy of each structure.
	energies         = []
	structure_params = []

	struct_idx        = 0
	current_struct_id = -1


	for struct_id in training_indices:

		current_structure = training_set.training_structures[struct_id]

		energies.append(current_structure[0].structure_energy)


		for atom in current_structure:
			structure_params.append(atom.structure_params)

	# Now we should have all of the training data ready, just not in PyTorch tensor format
	# quite yet.

	t_energies         = torch.tensor(np.transpose([energies])).type(torch.FloatTensor)
	t_structure_params = torch.tensor(structure_params).type(torch.FloatTensor)

	# During this phase of startup, we create the actual
	# PyTorch neural network objects as well as the optimizer
	# and any closure functions necessary for it.

	torch_net = TorchNet(neural_network_data, reduction_matrix=None)

	# Used to track the loss as a function of the iteration,
	# which will be dumped to a log at the end.
	

	# Now that the network and its inputs are setup, we produce a json file
	# with the inputs, and the outputs for that set of inputs.
	results = {}

	results["input"]  = structure_params
	results["output"] = torch_net(t_structure_params)

	return results


if __name__ == '__main__':

		Util.init(LOG_PATH)
		
