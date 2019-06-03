from sys import path
path.append("subroutines")

from   Config                 import *
from   TrainingSet            import TrainingSetFile
from   NeuralNetwork          import NeuralNetwork
from   PyTorchNet             import TorchNet
from   ConfigurationParser    import TrainingFileConfig
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import matplotlib.pyplot   as plt
import numpy               as np
import Util
from   Util import log, log_indent, log_unindent
from   time import time


if __name__ == '__main__':
	log("---------- Program Started ----------\n")

	# This will take the contents of the configuration file and
	# print them in a nice format in the log file. This process
	# does not keep comme.confignts that are in the configuration file.
	Util.LogConfiguration()

	log("Loading Data")
	log_indent()

	# The primary steps for loading are as follows:
	#     1) Load the neural network weights and biases.
	#     2) Load the LSParam file that contains Gi's and
	#        DFT energies for each structure.

	neural_network_data = NeuralNetwork(NEURAL_NETWORK_FILE) 
	training_set        = TrainingSetFile(TRAINING_SET_FILE)

	# Make sure that the configurations actually match.
	if neural_network_data.config != training_set.config:
		# Log both configurations so the user can more easily discern the problem.
		log("ERROR, CONFIGURATION MISMATCH")
		log("Comparison")
		log_indent()
		log("Neural Network File:")
		log_indent()
		log(str(neural_network_data.config))
		log_unindent()
		log("Training Data File:")
		log_indent()
		log(str(training_set.config))
		log_unindent()
		Util.cleanup()
		raise ValueError("The training data and neural network files have different configurations. See %s"%LOG_PATH)

	log_unindent()


	log("Configuring Training Data")
	log_indent()

	# During this phase of startup, we convert the raw training
	# data that was loaded into PyTorch tensors in the correct 
	# format for fast training. 

	# Generate the following arrays:
	#     1) An array with one correct energy value for each structure
	#     2) An array with a volume value for each structure
	#     3) An array that contains 1 / Number of Atoms for each structure
	#     4) A vector with the appropriate group weight for each structure
	#     5) A flattened vector containing a list of structure parameters 
	#        for each atom.
	#     6) A matrix that reduces a set of energy outputs for each atom
	#        into a list of total energies, one for each structure.

	# Randomly select a set of structure IDs to use as the training set
	# and use the rest as a validation set.
	n_pick             = int(TRAIN_TO_TOTAL_RATIO * training_set.n_structures)
	training_indices   = list(np.random.choice(training_set.n_structures, n_pick, replace=False))
	n_training_indices = len(training_indices)

	# We need to know how many atoms are part of the training set and 
	# how many are part of the validation set.
	n_train_atoms = 0
	for index in training_indices:
		n_train_atoms += len(training_set.training_structures[index])

	# The following code assumes that the values in the LSParam file are
	# ordered sequentially, first by structure, then by atom.

	# This reduction matrix will be multiplied by the output column vector
	# to reduce the energy of each atom to the energy of each structure.
	reduction_matrix = np.zeros((n_training_indices, n_train_atoms))
	energies         = []
	volumes          = []
	inverse_n_atoms  = []
	group_weights    = []
	structure_params = []

	idx               = 0
	struct_idx        = 0
	current_struct_id = -1
	is_training       = True

	# The following four variables keep track of the current row and
	# column in the reduction matrix that we need to be setting to 1.
	train_reduction_row    = -1
	train_reduction_column = 0

	for struct_id in training_indices:
		train_reduction_row += 1

		current_structure = training_set.training_structures[struct_id]

		energies.append(current_structure[0].structure_energy)
		volumes.append(current_structure[0].structure_volume)
		inverse_n_atoms.append(1.0 / current_structure[0].structure_n_atoms)

		# Select the appropriate weight based on the group,
		# if specified.
		if current_structure[0].group_name in WEIGHTS.keys():
			group_weights.append(WEIGHTS[current_structure[0].group_name])
		else:
			group_weights.append(1.0)

		for atom in current_structure:
			structure_params.append(atom.structure_params)
			reduction_matrix[train_reduction_row][train_reduction_column] = 1.0
			train_reduction_column += 1

	# Now we should have all of the training data ready, just not in PyTorch tensor format
	# quite yet.
	# TODO: Write code that generates the accompanying validation data and actually checks
	#       against it.

	energies         = torch.tensor(np.transpose([energies])).type(torch.FloatTensor)
	inverse_n_atoms  = torch.tensor(np.transpose([inverse_n_atoms])).type(torch.FloatTensor)
	group_weights    = torch.tensor(np.transpose([group_weights])).type(torch.FloatTensor)
	structure_params = torch.tensor(structure_params).type(torch.FloatTensor)
	reduction_matrix = torch.tensor(reduction_matrix).type(torch.FloatTensor)



	log_unindent()


	log("Configuring Neural Network")
	log_indent()

	# During this phase of startup, we create the actual
	# PyTorch neural network objects as well as the optimizer
	# and any closure functions necessary for it.

	torch_net = TorchNet(neural_network_data, reduction_matrix)

	optimizer = None
	if OPTIMIZATION_ALGORITHM == 'LBFGS':
		optimizer = optim.LBFGS(torch_net.get_parameters(), lr=LEARNING_RATE, max_iter=MAX_LBFGS_ITERATIONS)
	else:
		# TODO: Figure out if this should also be a configuration value.
		optimizer = optim.SGD(torch_net.get_parameters(), lr=0.001, momentum=0.9)


	# Used to track the loss as a function of the iteration,
	# which will be dumped to a log at the end.
	loss_values = [0.0]*MAXIMUM_TRAINING_ITERATIONS
	global last_loss
	last_loss   = 0.0

	def get_loss():
		global last_loss
		
		calculated_values = torch_net(structure_params)
		#print(calculated_values[:4])

		# Here we are multiplying each structure energy error (as calculated by the neural network),
		# by the reciprocal of the number of atoms in the structure. This is so that we are effectively
		# calculating the error per atom, not the error per structure.
		difference = torch.mul(calculated_values - energies, inverse_n_atoms)
		RMSE       = torch.sqrt((group_weights * (difference**2)).sum() / n_training_indices)
		last_loss  = RMSE.item()
		return RMSE

	# This is called by the optimizer in order to actually evaluate the
	# neural net and to calculate the error.
	def closure():
		optimizer.zero_grad()
		RMSE = get_loss()
		RMSE.backward()

		return RMSE


	def log_energy_vs_volume():
		f = open(E_VS_V_FILE, 'a')
		f.write(' '.join([str(v) for v in volumes]))
		f.write(' ')
		with torch.no_grad():
			values = [i.item() for i in torch_net(structure_params)]
		f.write(' '.join([str(e) for e in values]))
		f.write('\n')
		f.close()

	log_unindent()



	log("Beginning Training")
	log_indent()

	# Here we move everything to the CUDA device if necessary
	# and then begin the actual training loop. 
	# TODO: Write code that writes detailed training information
	#       to a JSON file with an accompanying program that 
	#       displays progress.


	current_iteration = 0
	start_time        = time()

	with torch.no_grad():
		# This will populate the list of loss values
		# with the initial value before training.
		get_loss()
	
	while current_iteration < MAXIMUM_TRAINING_ITERATIONS:
		loss_values[current_iteration] = last_loss

		if current_iteration % PROGRESS_INTERVAL == 0:
			print('Training Iteration = %5i | Error = %E'%(current_iteration, last_loss))

		if current_iteration % E_VS_V_INTERVAL == 0:
			# Add the current E vs. V values into the log file.
			log_energy_vs_volume()

		if OPTIMIZATION_ALGORITHM == 'LBFGS':
			optimizer.step(closure)
		else:
			optimizer.zero_grad()
			loss = get_loss()
			loss.backward()
			optimizer.step()

		

		current_iteration += 1

	end_time = time()
	print("Training Finished")
	print("Time Elapsed: %.1fs"%(end_time - start_time))

	log_unindent()

	log("Writing Output Files")
	log_indent()

	# Here we write the output neural network file and 
	# any log or assorted information files that the 
	# program generates.

	neural_network_data.layers = torch_net.get_network_values()
	neural_network_data.writeNetwork(NEURAL_NETWORK_SAVE_FILE)

	# Write the loss for every iteration into a file, 
	# separated by newline characters.

	loss_file = open(LOSS_LOG_PATH, 'w')
	for loss in loss_values:
		loss_file.write('%10.10E\n'%(loss))
	loss_file.close()

	log_unindent()


	Util.cleanup()