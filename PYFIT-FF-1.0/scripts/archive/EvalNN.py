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

def RunNetwork(nn_file, training_set, all_params):
	Util.init('garbage.txt')
	# The primary steps for loading are as follows:
	#     1) Load the neural network weights and biases.
	#     2) Load the LSParam file that contains Gi's and
	#        DFT energies for each structure.

	neural_network_data = NeuralNetwork(nn_file) 
	t_structure_params  = torch.tensor(np.transpose(all_params), requires_grad=False).type(torch.FloatTensor)
	#t_structure_params = torch.tensor(structure_params).type(torch.FloatTensor)

	# During this phase of startup, we create the actual
	# PyTorch neural network objects as well as the optimizer
	# and any closure functions necessary for it.
	with torch.no_grad():
		torch_net = TorchNet(neural_network_data, reduction_matrix=None, only_eval=True)

		# Used to track the loss as a function of the iteration,
		# which will be dumped to a log at the end.
		

		# Now that the network and its inputs are setup, we produce a json file
		# with the inputs, and the outputs for that set of inputs.
		results = {}

		results["output"] = np.transpose(torch_net(t_structure_params).numpy())[0].tolist()

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