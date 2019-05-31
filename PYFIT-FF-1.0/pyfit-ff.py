from sys import path
path.append("subroutines")

from   Config                 import *
from   TrainingSet            import TrainingSetFile
from   NeuralNetwork          import NeuralNetwork
from   PyTorchNet             import TorchNet
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
	# does not keep comments that are in the configuration file.
	Util.LogConfiguration()

	log("Loading Data")
	log_indent()

	# The primary steps for loading are as follows:
	#     1) Load the neural network weights and biases.
	#     2) Load the LSParam file that contains Gi's and
	#        DFT energies for each structure.

	neural_network_data = NeuralNetwork(NEURAL_NETWORK_FILE) 
	training_set        = TrainingSetFile(TRAINING_SET_FILE)

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
	training_indices   = list(range(training_set.n_structures))
	validation_indices = [i for i in range(training_set.n_structures) if i not in training_indices]

	n_training_indices = len(training_indices)

	# We need to know how many atoms are part of the training set and 
	# how many are part of the validation set.
	n_train_atoms = 0
	n_val_atoms   = 0

	for atom in training_set.training_inputs:
		if atom.structure_id in training_indices:
			n_train_atoms += 1
		else:
			n_val_atoms += 1

	# The following code assumes that the values in the LSParam file are
	# ordered sequentially, first by structure, then by atom.

	# This reduction matrix will be multiplied by the output column vector
	# to reduce the energy of each atom to the energy of each structure.
	reduction_matrix = np.zeros((len(training_indices), n_train_atoms))
	energies         = []
	volumes          = []
	inverse_n_atoms  = []
	group_weights    = []
	structure_params = []

	val_reduction_matrix = np.zeros((len(validation_indices), n_val_atoms))
	val_energies         = []
	val_volumes          = []
	val_inverse_n_atoms  = []
	val_group_weights    = []
	val_structure_params = []

	idx               = 0
	current_struct_id = -1
	is_training       = True

	# The following four variables keep track of the current row and
	# column in the reduction matrix that we need to be setting to 1.
	train_reduction_row    = -1
	train_reduction_column = 0
	val_reduction_row      = -1
	val_reduction_column   = 0

	while idx < training_set.n_atoms:
		# We take the first instance of each structure
		# ID and add the corresponding information into
		# the per-structure-id arrays and we add each 
		# set of struture parameters (per atom) into the 
		# structure_params array.
		current_input = training_set.training_inputs[idx]

		# Make sure that this is part of the training set.
		is_training = current_input.structure_id in training_indices

		if current_input.structure_id != current_struct_id:
			# We haven't processed this struct ID yet, do it now.
			# Also set this so we don't reprocess it.
			current_struct_id = current_input.structure_id

			# Since we just switched to a new structure, increment
			# the row in the corresponding reduction matrix that we
			# are setting.
			if is_training:
				train_reduction_row += 1
			else:
				val_reduction_row += 1

			(energies if is_training else val_energies).append(current_input.structure_energy)
			(volumes if is_training else val_volumes).append(current_input.structure_volume)
			(inverse_n_atoms if is_training else val_inverse_n_atoms).append(1.0 / current_input.structure_n_atoms)

			# Select the appropriate weight based on the group,
			# if specified.
			if current_input.group_name in WEIGHTS.keys():
				(group_weights if is_training else val_group_weights).append(WEIGHTS[current_input.group_name])
			else:
				(group_weights if is_training else val_group_weights).append(1.0)

		# We add the structure params regardless of whether or not
		# we have already processed a member of this structure.
		
		(structure_params if is_training else val_structure_params).append(current_input.structure_params)

		if is_training:
			reduction_matrix[train_reduction_row][train_reduction_column] = 1.0
			train_reduction_column += 1
		else:
			val_reduction_matrix[val_reduction_row][val_reduction_column] = 1.0
			val_reduction_column += 1


		idx += 1



	# Now we should have all of the training data ready, just not in PyTorch tensor format
	# quite yet.
	# TODO: Go further with the code that handles validation data set.

	energies         = torch.tensor(np.transpose([energies]), requires_grad=True).type(torch.FloatTensor)
	inverse_n_atoms  = torch.tensor(np.transpose([inverse_n_atoms]), requires_grad=True).type(torch.FloatTensor)
	group_weights    = torch.tensor(np.transpose([group_weights]), requires_grad=True).type(torch.FloatTensor)
	structure_params = torch.tensor(structure_params, requires_grad=True).type(torch.FloatTensor)
	reduction_matrix = torch.tensor(reduction_matrix, requires_grad=True).type(torch.FloatTensor)

	# The dataset is now ready, minus the validation part (TODO).


	log_unindent()


	log("Configuring Neural Network")
	log_indent()

	# During this phase of startup, we create the actual
	# PyTorch neural network objects as well as the optimizer
	# and any closure functions necessary for it.

	torch_net = TorchNet(neural_network_data, reduction_matrix)

	optimizer = None
	if OPTIMIZATION_ALGORITHM == 'LBFGS':
		optimizer = optim.LBFGS(torch_net.get_parameters(), lr=LEARNING_RATE, max_iter=10)
	else:
		# TODO: Figure out if this should also be a configuration value.
		optimizer = optim.SGD(torch_net.get_parameters(), lr=0.001, momentum=0.9)

	# This is called by the optimizer in order to actually evaluate the
	# neural net and to calculate the error.
	def closure():

		if OPTIMIZATION_ALGORITHM == 'LBFGS':
			optimizer.zero_grad()

		
		calculated_values = torch_net(structure_params)

		# Here we are multiplying each structure energy error (as calculated by the neural network),
		# by the reciprocal of the number of atoms in the structure. This is so that we are effectively
		# calculating the error per atom, not the error per structure.
		difference = torch.mul(calculated_values - energies, inverse_n_atoms)
		RMSE       = torch.sqrt(torch.mul(group_weights, (difference**2)).sum() / n_training_indices)
		
		print("%i %E"%(current_iteration, RMSE.item()))
		if OPTIMIZATION_ALGORITHM == 'LBFGS':
			RMSE.backward()
		

		return RMSE



	log_unindent()



	log("Beginning Training")
	log_indent()

	# Here we move everything to the CUDA device if necessary
	# and then begin the actual training loop. 
	# TODO: Write code that writes detailed training information
	#       to a JSON file with an accompanying program that 
	#       displays progress.


	global current_iteration
	current_iteration = 1
	start_time        = time()
	while current_iteration <= MAXIMUM_TRAINING_ITERATIONS:
		if OPTIMIZATION_ALGORITHM == 'LBFGS':
			optimizer.step(closure)
		else:
			optimizer.zero_grad()
			loss = closure()
			loss.backward()
			optimizer.step()

		# if current_iteration % PROGRESS_INTERVAL == 0:
		# 	print("Iteration = %6i, RMSE = %E"%(current_iteration, current_error))

		# TODO: Implement automatic saving of the network at some
		#       interval in case there is a crash.
		#      
		#       Implement periodic storage of training progress so
		#       it can be graphed an analyzed.
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

	log_unindent()


	Util.cleanup()