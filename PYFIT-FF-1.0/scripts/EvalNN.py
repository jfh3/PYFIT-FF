from sys import path
path.append("subroutines")

from   TrainingSet   import TrainingSetFile
from   NeuralNetwork import NeuralNetwork
from   PyTorchNet    import TorchNet
import numpy as np
import torch
import Util
import sys
import os
import copy
import json

def GetTrainingSetInstance(file):
	Util.init('garbage.txt')
	f = TrainingSetFile(file)
	Util.cleanup()
	return f

def RunNetwork(nn_file, training_set):
	Util.init('garbage.txt')
	# The primary steps for loading are as follows:
	#     1) Load the neural network weights and biases.
	#     2) Load the LSParam file that contains Gi's and
	#        DFT energies for each structure.

	neural_network_data = NeuralNetwork(nn_file) 

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

	torch_net = TorchNet(neural_network_data, reduction_matrix=None, only_eval=True)

	# Used to track the loss as a function of the iteration,
	# which will be dumped to a log at the end.
	

	# Now that the network and its inputs are setup, we produce a json file
	# with the inputs, and the outputs for that set of inputs.
	results = {}

	results["input"]  = structure_params
	results["output"] = [i[0] for i in torch_net(t_structure_params).tolist()]

	Util.cleanup()

	return results


if __name__ == '__main__':
	# This program takes three arguments.
	#     1) The neural network file to use.
	#     2) The LSPARAM file to use.
	#     3) Where to write the output to.

	if len(sys.argv) != 4:
		eprint("This program takes 3 arguments.")
		sys.exit(1)

	nn_file      = sys.argv[1]
	lsparam_file = sys.argv[2]
	output_path  = sys.argv[3]

	


	lsparam_file = GetTrainingSetInstance(lsparam_file)
	results = RunNetwork(nn_file, lsparam_file)

	f = open(output_path, 'w')
	f.write(json.dumps(results))
	f.close()