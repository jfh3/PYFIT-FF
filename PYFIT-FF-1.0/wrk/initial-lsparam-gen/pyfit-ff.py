from sys import path
path.append("subroutines")

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

def ComputeStructureParameters():
	log("Beginning Structural Parameter Computation")
	log_indent()


	neural_network        = NeuralNetwork(NEURAL_NETWORK_FILE)
	poscar_data           = PoscarDataFile(POSCAR_DATA_FILE)
	neighborLists         = generateNeighborList(poscar_data, neural_network)
	structural_parameters = GenerateStructuralParameters(poscar_data, neighborLists, neural_network)
	

	WriteTrainingSet(
		LSPARAM_FILE, 
		neural_network.config, 
		poscar_data, 
		structural_parameters
	)

	if NEIGHBOR_FILE != None and NEIGHBOR_FILE != '':
		WriteTrainingSet(
			NEIGHBOR_FILE, 
			neural_network.config, 
			poscar_data, 
			structural_parameters,
			neighborLists
		)
	
	log_unindent()

def GraphError(graph_error, graph_val, annealings):
	plots = []
	names = []

	if graph_error:
		file  = open(LOSS_LOG_PATH, 'r')
		raw   = file.read()
		file.close()

		error_values = [float(i) for i in raw.split('\n') if not i.isspace() and i != '']
		e_indices      = range(len(error_values))
		plots.append(plt.scatter(e_indices, error_values, s=8))
		names.append("Training Error")

	if graph_val:
		file  = open(VALIDATION_LOG_PATH, 'r')
		raw   = file.read()
		file.close()

		validation_values = [float(i) for i in raw.split('\n') if not i.isspace() and i != '']
		val_indices       = np.array(range(len(validation_values))) * VALIDATION_INTERVAL
		plots.append(plt.scatter(val_indices, validation_values, s=8))
		names.append("Validation Error")

	if annealings != None:
		# Draw vertical red lines where randomization took place.
		for annealing in annealings:
			plt.axvline(annealing, c='red')

	plt.legend(plots, names)
	plt.xlabel("Training Iteration")
	plt.ylabel("Root Mean Squared Error")
	plt.title("Error vs. Iteration")
	plt.show()

def GraphGroupError(to_highlight):
	from mpldatacursor import datacursor
	import warnings
	warnings.filterwarnings("ignore")

	file = open(GROUP_ERROR_FILE, 'r')
	raw  = file.read()
	file.close()

	lines       = [l for l in raw.split('\n') if l != '']
	group_names = lines[0].split(' ')[1:]
	errors      = [[float(c) for c in l.split(' ')[1:]] for l in lines[1:]]
	iterations  = [GROUP_ERROR_RECORD_INTERVAL*int(l.split(' ')[0]) for l in lines[1:]]

	plots = []
	for group in range(len(group_names)):
		group_errors = [e[group] for e in errors]

		if to_highlight != None and group_names[group] == to_highlight:
			plots.append(plt.scatter(iterations, group_errors, s=24, label=group_names[group], marker="x", c='red', zorder=100000))
		else:
			plots.append(plt.scatter(iterations, group_errors, s=4, label=group_names[group]))

		

	datacursor(formatter='{label}'.format)

	plt.xlabel("Iteration")
	plt.ylabel("Group Error")

	plt.title("Error vs. Iteration")
	plt.show()

def CompareStructureParameters(first, second):
	first_file  = TrainingSetFile(first)
	second_file = TrainingSetFile(second)

	if first_file.config.layer_sizes[0] != second_file.config.layer_sizes[0]:
		print("The files have a different number of structure parameters per atom.")
		exit()

	if first_file.n_structures != second_file.n_structures:
		print("The two files have a different number of structures.")
		exit()

	if first_file.n_atoms != second_file.n_atoms:
		print("The files have a different number of atoms.")
		exit()

	if first_file.config != second_file.config:
		print("The two files have different configuration parameters.")
		print("You shouldn't expect their structural parameters to match.\n")

	# Now we want a flattened array of all structural parameters in order from
	# both files.
	first  = []
	second = []

	for struct_idx in range(len(first_file.training_structures.keys())):
		for atom_idx in range(len(first_file.training_structures[struct_idx])):
			for gi_idx in range(len(first_file.training_structures[struct_idx][atom_idx].structure_params)):
				first.append(first_file.training_structures[struct_idx][atom_idx].structure_params[gi_idx])
				second.append(second_file.training_structures[struct_idx][atom_idx].structure_params[gi_idx])


	first  = np.array(first)
	second = np.array(second)
	percent_error = np.abs(np.abs(first - second) / first)

	total_error = np.sum(percent_error)
	std_error   = np.std(percent_error)
	mean_error  = np.mean(percent_error)
	max_error   = percent_error.max()
	min_error   = percent_error.min()

	print("Error Summary")
	print("\ttotal: %e"%total_error)
	print("\tmean:  %e"%mean_error)
	print("\tstd:   %e"%std_error)
	print("\tmax:   %e"%max_error)
	print("\tmin:   %e"%min_error)

	plt.title("Structure Parameter Comparison")
	plt.scatter(first[:100000], second[:100000], s=3, c='red')
	plt.plot([min(first), max(first)], [min(first), max(first)])
	plt.xlabel("First File")
	plt.ylabel("Second File")
	plt.show()

def TrainNetwork(force_cpu, randomize_nn, gpu_affinity=0):
	log("Beginning Training Process")
	log_indent()

	device = torch.device("cuda:%i"%(gpu_affinity) if torch.cuda.is_available() else "cpu")
	if force_cpu:
		device = 'cpu'
		
	log("Loading Data")
	log_indent()

	# The primary steps for loading are as follows:
	#     1) Load the neural network weights and biases.
	#     2) Load the LSParam file that contains Gi's and
	#        DFT energies for each structure.

	neural_network_data = NeuralNetwork(NEURAL_NETWORK_FILE) 
	training_set        = TrainingSetFile(TRAINING_SET_FILE)

	if randomize_nn:
		neural_network_data.generateNetwork(just_randomize=True)

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

	
	# DEBUG
	# This should ensure that all validation/training splits
	# come out the same way.
	np.random.seed(1)


	# It is infinitely faster to shuffle the array and then pick the first n_pick
	# indices rather than using np.random.choice. If we do the latter, determining
	# the precise list of validation indices entails collecting all indices not in
	# the training set, which takes n_structures^2 time. This is because for each 
	# indice, we have to scan the whole list of training indices to verify that it
	# isn't in there.
	if OBJECTIVE_FUNCTION == 'group-targets' or GROUP_WISE_VALIDATION_SPLIT:
		# If we are training to specific group error targets, we
		# need to split up the training and validation data so that
		# each group is evenly represented in both the training and
		# validation set. If we don't do this, there is a random 
		# chance that a group that we want to train to a specific
		# error will be underrepresented in the training set or
		# underrepresented in the validation set.

		# Split up the training structures by their group.
		training_group_dict = {}
		current_group       = training_set.training_structures[0][0].group_name
		training_group_dict[current_group] = []
		for struct_idx in training_set.training_structures:
			current_group = training_set.training_structures[struct_idx][0].group_name
			if current_group not in training_group_dict:
				training_group_dict[current_group] = []

			# Here we are appending a tuple structured as (structure, index within list)
			# This is so that the index within the large list of structures is not
			# lost while constructing the grouping.
			training_group_dict[current_group].append(struct_idx)

		# Now that we have all of the structural groups separated, we select 
		# the appropriate percentage of each group and add it into the list of
		# training and validation indices.
		training_indices   = []
		validation_indices = []

		n_iter = 0

		for group in training_group_dict:
			n_pick      = int(TRAIN_TO_TOTAL_RATIO * len(training_group_dict[group]))
			all_indices = np.array(training_group_dict[group])

			indices_and_volumes = [(struct, training_set.training_structures[struct][0].structure_volume) for struct in training_group_dict[group]]
			sorted_by_volume    = sorted(indices_and_volumes, key=lambda x: x[1])
			indices_ordered     = np.array([i[0] for i in sorted_by_volume])
			all_indices         = indices_ordered[1:-1]

			np.random.shuffle(all_indices)

			all_indices = list(all_indices)

			training_indices.extend(list(all_indices[:n_pick]))
			validation_indices.extend(list(all_indices[n_pick:]))

			training_indices.extend([indices_ordered[0], indices_ordered[-1]])

		n_training_indices   = len(training_indices)
		n_validation_indices = len(validation_indices)

	else:
		# Randomly select a set of structure IDs to use as the training set
		# and use the rest as a validation set.
		n_pick               = int(TRAIN_TO_TOTAL_RATIO * training_set.n_structures)
		all_indices          = np.array(range(training_set.n_structures))
		np.random.shuffle(all_indices)
		all_indices          = list(all_indices)
		training_indices     = list(all_indices[:n_pick])
		validation_indices   = list(all_indices[n_pick:])
		n_training_indices   = len(training_indices)
		n_validation_indices = len(validation_indices)
	

	# We need to know how many atoms are part of the training set and 
	# how many are part of the validation set.
	n_train_atoms = 0
	for index in training_indices:
		n_train_atoms += len(training_set.training_structures[index])

	# The following code assumes that the values in the LSParam file are
	# ordered sequentially, first by structure, then by atom.

	if OBJECTIVE_FUNCTION == 'group-targets':
		# If we are targeting specific errors for different groups,
		# we need to construct a matrix for determining the error
		# of each group. This is very similar to the regular reduction
		# matrix.

		# We need to know the number of distinct groups.
		n_groups = 1

		last_group  = training_set.training_structures[training_indices[0]][0].group_name
		used_groups = [last_group]
		for struct_id in training_indices:
			current_group = training_set.training_structures[struct_id][0].group_name
			if current_group != last_group:
				if current_group not in used_groups:
					n_groups  += 1
					last_group = current_group
					used_groups.append(current_group)

		group_reduction_matrix = np.zeros((n_groups, n_training_indices))

		# We also need an array of group target error values in the same
		# order as the groups in the output of the group reduction matrix
		# so that we can compute the individual groups errors with a PyTorch
		# operation.
		group_target_array     = []

		# We need to know the size of each group so that we can divide by this
		# when calculating the RMSE for each group.
		group_sizes            = [0]*n_groups

	# This reduction matrix will be multiplied by the output column vector
	# to reduce the energy of each atom to the energy of each structure.
	reduction_matrix = np.zeros((n_training_indices, n_train_atoms))
	energies         = []
	volumes          = []
	inverse_n_atoms  = []
	group_weights    = []
	structure_params = []

	struct_idx        = 0
	current_struct_id = -1

	# The following four variables keep track of the current row and
	# column in the reduction matrix that we need to be setting to 1.
	train_reduction_row    = -1
	train_reduction_column = 0

	if OBJECTIVE_FUNCTION == 'group-targets':
		# Here we store a row index that corresponds to each
		# structural group in the group_reduction_matrix.
		group_rows             = {}
		current_group_row_idx  = 1
		current_group_row      = 0
		current_group_column   = 0
		last_group             = training_set.training_structures[training_indices[0]][0].group_name
		group_rows[last_group] = 0

		# This is just used for reporting per group error at the end.
		ordered_group_names    = [last_group]

		# Here we handle the insertion of the target error value
		# for this structural group into the array.
		if last_group in SUBGROUP_TARGETS:
			group_target_array.append(SUBGROUP_TARGETS[last_group])
		else:
			group_target_array.append(DEFAULT_TARGET)

	for struct_id in training_indices:

		train_reduction_row += 1

		current_structure = training_set.training_structures[struct_id]

		if OBJECTIVE_FUNCTION == 'group-targets':
			# Figure out what group the current structure corresponds to.
			current_group = current_structure[0].group_name

			# If we have switched groups, some things need to be handled.
			if current_group != last_group:

				# Make sure that a row has been assigned for this group.
				if current_group not in group_rows:
					group_rows[current_group] = current_group_row_idx
					current_group_row_idx    += 1
					ordered_group_names.append(current_group)

					if current_group in SUBGROUP_TARGETS:
						group_target_array.append(SUBGROUP_TARGETS[current_group])
					else:
						group_target_array.append(DEFAULT_TARGET)

				current_group_row = group_rows[current_group]
				last_group        = current_group


			group_sizes[current_group_row] += 1

			# Set the appropriate column of the
			# appropriate row for this group and structure.
			group_reduction_matrix[current_group_row][current_group_column] = 1.0
			current_group_column += 1

		energies.append(current_structure[0].structure_energy)
		volumes.append(current_structure[0].structure_volume)
		inverse_n_atoms.append(1.0 / current_structure[0].structure_n_atoms)

		# Select the appropriate weight based on the group,
		# if specified.
		if current_structure[0].group_name in WEIGHTS.keys():
			group_weights.append(WEIGHTS[current_structure[0].group_name])
		else:
			group_weights.append(DEFAULT_WEIGHT)

		for atom in current_structure:
			structure_params.append(atom.structure_params)
			reduction_matrix[train_reduction_row][train_reduction_column] = 1.0
			train_reduction_column += 1

	if OBJECTIVE_FUNCTION == 'group-targets':
		group_reduction_matrix = torch.tensor(group_reduction_matrix).type(torch.FloatTensor).to(device)
		group_target_array     = torch.tensor(np.transpose([group_target_array])).type(torch.FloatTensor).to(device)
		group_sizes            = torch.tensor(np.transpose([group_sizes])).type(torch.FloatTensor).to(device)

	# Now we should have all of the training data ready, just not in PyTorch tensor format
	# quite yet.

	energies           = torch.tensor(np.transpose([energies])).type(torch.FloatTensor).to(device)
	inverse_n_atoms    = torch.tensor(np.transpose([inverse_n_atoms])).type(torch.FloatTensor).to(device)
	group_weights      = torch.tensor(np.transpose([group_weights])).type(torch.FloatTensor).to(device)
	structure_params   = torch.tensor(structure_params).type(torch.FloatTensor).to(device)
	n_training_indices = torch.tensor(n_training_indices).type(torch.FloatTensor).to(device)
	reduction_matrix = torch.tensor(reduction_matrix).type(torch.FloatTensor)
	# The reduction matrix will be automatically moved to the device (cpu or cuda) when
	# torch_net.to(device) is called later.

	# Now we do essentially the same thing as the previous lines, except for the validation
	# dataset.

	if TRAIN_TO_TOTAL_RATIO != 1.0:
		n_val_atoms = 0
		for index in validation_indices:
			n_val_atoms += len(training_set.training_structures[index])

		if OBJECTIVE_FUNCTION == 'group-targets':
			val_n_groups    = 1
			val_last_group  = training_set.training_structures[validation_indices[0]][0].group_name
			val_used_groups = [val_last_group]
			for struct_id in validation_indices:
				current_group = training_set.training_structures[struct_id][0].group_name
				if current_group != val_last_group:
					if current_group not in val_used_groups:
						val_n_groups  += 1
						val_last_group = current_group
						val_used_groups.append(current_group)

			val_group_reduction_matrix = np.zeros((val_n_groups, n_validation_indices))

			# We also need an array of group target error values in the same
			# order as the groups in the output of the group reduction matrix
			# so that we can compute the individual groups errors with a PyTorch
			# operation.
			val_group_target_array     = []

			# We need to know the size of each group so that we can divide by this
			# when calculating the RMSE for each group.
			val_group_sizes            = [0]*val_n_groups

			# Here we store a row index that corresponds to each
			# structural group in the group_reduction_matrix.
			val_group_rows             = {}
			val_current_group_row_idx  = 1
			val_current_group_row      = 0
			val_current_group_column   = 0
			val_last_group             = training_set.training_structures[validation_indices[0]][0].group_name
			val_group_rows[val_last_group] = 0

			val_ordered_group_names = [val_last_group]

			# Here we handle the insertion of the target error value
			# for this structural group into the array.
			if val_last_group in SUBGROUP_TARGETS:
				val_group_target_array.append(SUBGROUP_TARGETS[val_last_group])
			else:
				val_group_target_array.append(DEFAULT_TARGET)

		# The following code assumes that the values in the LSParam file are
		# ordered sequentially, first by structure, then by atom.

		# This reduction matrix will be multiplied by the output column vector
		# to reduce the energy of each atom to the energy of each structure.
		val_reduction_matrix = np.zeros((n_validation_indices, n_val_atoms))
		val_energies         = []
		val_volumes          = []
		val_inverse_n_atoms  = []
		val_group_weights    = []
		val_structure_params = []

		val_struct_idx        = 0
		val_current_struct_id = -1

		# The following four variables keep track of the current row and
		# column in the reduction matrix that we need to be setting to 1.
		val_reduction_row    = -1
		val_reduction_column = 0


		for struct_id in validation_indices:

			val_reduction_row += 1

			current_structure = training_set.training_structures[struct_id]

			if OBJECTIVE_FUNCTION == 'group-targets':
				# Figure out what group the current structure corresponds to.
				current_group = current_structure[0].group_name

				# If we have switched groups, some things need to be handled.
				if current_group != val_last_group:

					# Make sure that a row has been assigned for this group.
					if current_group not in val_group_rows:
						val_group_rows[current_group] = val_current_group_row_idx
						val_current_group_row_idx    += 1
						val_ordered_group_names.append(current_group)

						if current_group in SUBGROUP_TARGETS:
							val_group_target_array.append(SUBGROUP_TARGETS[current_group])
						else:
							val_group_target_array.append(DEFAULT_TARGET)

					val_current_group_row = val_group_rows[current_group]
					val_last_group        = current_group


				val_group_sizes[val_current_group_row] += 1

				# Set the appropriate column of the
				# appropriate row for this group and structure.
				val_group_reduction_matrix[val_current_group_row][val_current_group_column] = 1.0
				val_current_group_column += 1

			val_energies.append(current_structure[0].structure_energy)
			val_volumes.append(current_structure[0].structure_volume)
			val_inverse_n_atoms.append(1.0 / current_structure[0].structure_n_atoms)

			# Select the appropriate weight based on the group,
			# if specified.
			if current_structure[0].group_name in WEIGHTS.keys():
				val_group_weights.append(WEIGHTS[current_structure[0].group_name])
			else:
				val_group_weights.append(DEFAULT_WEIGHT)

			for atom in current_structure:
				val_structure_params.append(atom.structure_params)
				val_reduction_matrix[val_reduction_row][val_reduction_column] = 1.0
				val_reduction_column += 1



		if OBJECTIVE_FUNCTION == 'group-targets':
			val_group_reduction_matrix = torch.tensor(val_group_reduction_matrix).type(torch.FloatTensor)
			val_group_target_array     = torch.tensor(np.transpose([val_group_target_array])).type(torch.FloatTensor)
			val_group_sizes            = torch.tensor(np.transpose([val_group_sizes])).type(torch.FloatTensor)

		# Now we should have all of the training data ready, just not in PyTorch tensor format
		# quite yet.

		val_energies         = torch.tensor(np.transpose([val_energies])).type(torch.FloatTensor)
		val_inverse_n_atoms  = torch.tensor(np.transpose([val_inverse_n_atoms])).type(torch.FloatTensor)
		val_group_weights    = torch.tensor(np.transpose([val_group_weights])).type(torch.FloatTensor)
		val_structure_params = torch.tensor(val_structure_params).type(torch.FloatTensor)
		val_reduction_matrix = torch.tensor(val_reduction_matrix).type(torch.FloatTensor)
		# This stays on the cpu because thats the only place where we ever compute validation error.

	log_unindent()

	if OBJECTIVE_FUNCTION == 'group-targets':
		if UNWEIGHTED_NEGATIVE_ERROR:
			# This is compared to the group error - target group error
			# with a max function in order to score error below the
			# target error as zero.
			negative_error_zero_compare = torch.zeros(len(group_sizes)).type(torch.FloatTensor).to(device)

	log("Configuring Neural Network")
	log_indent()

	# During this phase of startup, we create the actual
	# PyTorch neural network objects as well as the optimizer
	# and any closure functions necessary for it.

	torch_net = TorchNet(neural_network_data, reduction_matrix).to(device)

	optimizer = None
	if OPTIMIZATION_ALGORITHM == 'LBFGS':
		optimizer = optim.LBFGS(torch_net.get_parameters(), lr=LEARNING_RATE, max_iter=MAX_LBFGS_ITERATIONS)
	else:
		# TODO: Figure out if this should also be a configuration value.
		optimizer = optim.SGD(torch_net.get_parameters(), lr=LEARNING_RATE, momentum=0.9)


	if OBJECTIVE_FUNCTION == 'group-targets':
		group_loss_values = [None]*(MAXIMUM_TRAINING_ITERATIONS // GROUP_ERROR_RECORD_INTERVAL)


		def get_group_error():
			with torch.no_grad():
				temp_net          = copy.deepcopy(torch_net)
				calculated_values = temp_net(structure_params)
				difference        = torch.mul(calculated_values - energies, inverse_n_atoms)
				# Multiple by the group reduction matrix in order to get the 
				# error in each group.

				# This multiplication sums the difference squared on a per-group basis.
				per_group_diff_squared_sum = group_reduction_matrix.mm(difference**2)
				per_group_diff_squared_avg = per_group_diff_squared_sum / group_sizes
				per_group_rmse             = torch.sqrt(per_group_diff_squared_avg)
			return [rmse[0] for rmse in per_group_rmse.cpu().tolist()]

	# Used to track the loss as a function of the iteration,
	# which will be dumped to a log at the end.
	loss_values            = [None]*MAXIMUM_TRAINING_ITERATIONS
	validation_loss_values = []
	global last_loss
	last_loss   = 0.0

	if TRAIN_TO_TOTAL_RATIO != 1.0:
		if OBJECTIVE_FUNCTION == 'group-targets':
			if UNWEIGHTED_NEGATIVE_ERROR:
				def get_validation_loss():
					with torch.no_grad():
						temp_net          = copy.deepcopy(torch_net).cpu()
						temp_net.set_reduction_matrix(val_reduction_matrix)

						calculated_values = temp_net(val_structure_params)
						difference        = torch.mul(calculated_values - val_energies, val_inverse_n_atoms)
						# Multiple by the group reduction matrix in order to get the 
						# error in each group.

						# This multiplication sums the difference squared on a per-group basis.
						per_group_diff_squared_sum  = val_group_reduction_matrix.mm(difference**2)
						per_group_diff_squared_avg  = per_group_diff_squared_sum / val_group_sizes
						per_group_rmse              = torch.sqrt(per_group_diff_squared_avg)
						loss_zeroed                 = torch.max(per_group_rmse - val_group_target_array, negative_error_zero_compare.cpu())
						loss                        = (SUBGROUP_ERROR_COEFFICIENT*loss_zeroed**2).sum()
					return loss.cpu().item()
			else:
				def get_validation_loss():
					with torch.no_grad():
						temp_net          = copy.deepcopy(torch_net).cpu()
						temp_net.set_reduction_matrix(val_reduction_matrix)

						calculated_values = temp_net(val_structure_params)
						difference        = torch.mul(calculated_values - val_energies, val_inverse_n_atoms)
						# Multiple by the group reduction matrix in order to get the 
						# error in each group.

						# This multiplication sums the difference squared on a per-group basis.
						per_group_diff_squared_sum = val_group_reduction_matrix.mm(difference**2)
						per_group_diff_squared_avg = per_group_diff_squared_sum / val_group_sizes
						per_group_rmse             = torch.sqrt(per_group_diff_squared_avg)
						loss                       = (SUBGROUP_ERROR_COEFFICIENT*(val_group_target_array - per_group_rmse)**2).sum()
					return loss.cpu().item()
		else:
			def get_validation_loss():
				with torch.no_grad():
					temp_net          = copy.deepcopy(torch_net).cpu()
					temp_net.set_reduction_matrix(val_reduction_matrix)

					calculated_values = temp_net(val_structure_params)
					difference = torch.mul(calculated_values - val_energies, val_inverse_n_atoms)
					RMSE       = torch.sqrt((val_group_weights * (difference**2)).sum() / n_validation_indices)
					RMSE       = RMSE.cpu().item()
				return RMSE

	if OBJECTIVE_FUNCTION == 'group-targets':
		if UNWEIGHTED_NEGATIVE_ERROR:
			def get_loss():
				global last_loss
				
				calculated_values = torch_net(structure_params)
				difference        = torch.mul(calculated_values - energies, inverse_n_atoms)
				# Multiple by the group reduction matrix in order to get the 
				# error in each group.

				# This multiplication sums the difference squared on a per-group basis.
				per_group_diff_squared_sum = group_reduction_matrix.mm(difference**2)
				per_group_diff_squared_avg = per_group_diff_squared_sum / group_sizes
				per_group_rmse             = torch.sqrt(per_group_diff_squared_avg)
				loss_zeroed                = torch.max(per_group_rmse - group_target_array, negative_error_zero_compare)
				loss                       = (SUBGROUP_ERROR_COEFFICIENT*loss_zeroed**2).sum()
				last_loss  = loss.cpu().item()
				return loss
		else:
			def get_loss():
				global last_loss
				
				calculated_values = torch_net(structure_params)
				difference        = torch.mul(calculated_values - energies, inverse_n_atoms)
				# Multiple by the group reduction matrix in order to get the 
				# error in each group.

				# This multiplication sums the difference squared on a per-group basis.
				per_group_diff_squared_sum = group_reduction_matrix.mm(difference**2)
				per_group_diff_squared_avg = per_group_diff_squared_sum / group_sizes
				per_group_rmse             = torch.sqrt(per_group_diff_squared_avg)
				loss                       = ((group_target_array - per_group_rmse)**2).sum()
				last_loss  = loss.cpu().item()
				return loss
	else:
		def get_loss():
			global last_loss


			# print('----------------------------------------\n')
			# torch_net.print_device_info()
			# print("Params: %s"%(structure_params.device))
			# print("Energy: %s"%(energies.device))
			# print("Inverse: %s"%(inverse_n_atoms.device))
			# print("GW: %s"%(group_weights.device))
			# print("N: %s"%(n_training_indices.device))
			# print('----------------------------------------\n')

			calculated_values = torch_net(structure_params)

			# Here we are multiplying each structure energy error (as calculated by the neural network),
			# by the reciprocal of the number of atoms in the structure. This is so that we are effectively
			# calculating the error per atom, not the error per structure.
			difference = torch.mul(calculated_values - energies, inverse_n_atoms)
			RMSE       = torch.sqrt((group_weights * (difference**2)).sum() / n_training_indices)
			last_loss  = RMSE.cpu().item()
			return RMSE

	# This is called by the optimizer in order to actually evaluate the
	# neural net and to calculate the error.
	def closure():
		optimizer.zero_grad()
		RMSE = get_loss()
		RMSE.backward()

		return RMSE


	def log_energy_vs_volume():
		f = open(E_VS_V_FILE, 'a', FILE_BUFFERING)
		f.write(' '.join([str(v) for v in volumes]))
		f.write(' ')
		with torch.no_grad():
			temp   = torch_net(structure_params)
			values = [i.item() for i in temp.cpu()]
		f.write(' '.join([str(e) for e in values]))
		f.write('\n')
		f.close()

	def partial_randomize(network):
		# Here we copy the network to the CPU,
		# have it modify itself based on the 
		# parameter given by the user and then
		# copy it back to whichever device we
		# are training on.
		with torch.no_grad():
			net_cpu = network.cpu()
			net_cpu.randomize_self(PLATEAU_ANNEALING_RAND_STD)
			torch_net = net_cpu.to(device)

		global LEARNING_RATE
		LEARNING_RATE -= PLATEAU_ANNEALING_LR_DECREMENT 
		# Now that there have been copies and modifications,
		# generate an optimizer for the new parameters.
		if OPTIMIZATION_ALGORITHM == 'LBFGS':
			optimizer = optim.LBFGS(torch_net.get_parameters(), lr=LEARNING_RATE, max_iter=MAX_LBFGS_ITERATIONS)
		else:
			# TODO: Figure out if this should also be a configuration value.
			optimizer = optim.SGD(torch_net.get_parameters(), lr=LEARNING_RATE, momentum=0.9)

		return torch_net, optimizer


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

	# This keeps track of the number of times in a row that
	# the difference between the error of two subsequent iterations
	# has been below FLAT_ERROR_STOP. When this exceeds or equals 
	# FLAT_ERROR_ITERATIONS training will terminate.
	error_below_threshold_count = 0


	# These values are used to keep track of how many times
	# validation_error - training_error has increased in a row.
	# They are used to shutdown the training process if the 
	# network becomes overfit. See OVERFIT_ERROR_STOP and
	# OVERFIT_INCREASE_MAX_ITERATIONS in Config.py
	last_training_validation_difference  = 10.0
	train_validate_increase_count        = 0

	# This is used to keep track of the number of times that the
	# system has reached a plateau in the error and then been partially
	# randomized in and attempt to break out of a local minima.
	# See PLATEAU_ANNEALING_MAX_ITERATIONS and PLATEAU_ANNEALING_RAND_STD
	# in Config.py
	plateau_annealing_count = 0

	# We store the iterations that the randomization process
	# was performed on here, so that they can be displayed at
	# the end of requested.
	plateau_annealing_iterations = []

	# If the training gets interrupted by ctrl-c,
	# this will be set to instruct the program to exit
	# instead of performing other tasks asked for by the 
	# user.
	terminate_early = False

	with torch.no_grad():
		# This will populate the list of loss values
		# with the initial value before training.
		get_loss()
		log("Initial RMSE before any training: %E"%(last_loss))
	
	bar = ProgressBar("Training", 30, MAXIMUM_TRAINING_ITERATIONS + 1, update_every = PROGRESS_INTERVAL)

	while current_iteration < MAXIMUM_TRAINING_ITERATIONS:
		try:
			# Most of this code actually just handles stopping conditions and
			# error logging.

			loss_values[current_iteration] = last_loss

			bar.update(current_iteration + 1)

			if current_iteration % GROUP_ERROR_RECORD_INTERVAL == 0:
				if OBJECTIVE_FUNCTION == 'group-targets':
					group_loss_values[current_iteration // GROUP_ERROR_RECORD_INTERVAL] = get_group_error()

			# This if statement handles the logic to ensure that the training
			# stops if the validation error gets too far away from the training
			# error or if the validation error starts to consistently diverge 
			# from the training error.
			if current_iteration % VALIDATION_INTERVAL == 0 and TRAIN_TO_TOTAL_RATIO != 1.0:
				validation_loss_values.append(get_validation_loss())

				if current_iteration > CHECK_OVERFIT_AFTER:
					# Make sure that there isn't an overfit.
					train_val_diff = validation_loss_values[-1] - last_loss

					if train_val_diff >= OVERFIT_ERROR_STOP:

						# Iterations since last annealing
						if len(plateau_annealing_iterations) > 0:
							last_annealing_iteration = plateau_annealing_iterations[-1]
						else:
							last_annealing_iteration = -CHECK_OVERFIT_AFTER

						iterations_since_last_randomize = current_iteration - last_annealing_iteration
						# We want to makre sure that we don't check this too soon after an
						# annealing, because it will likely be high.

						if iterations_since_last_randomize > CHECK_OVERFIT_AFTER:
							log("Difference between validation error and training error was %f."%train_val_diff)
							bar.finish()
							print("The Network Appears to be Becoming Overfit.")
							break

					if train_val_diff > last_training_validation_difference and train_val_diff > 0.0:
						log("Iteration: %i, (Validation Error - Training Error) Increase = %f"%(current_iteration, train_val_diff))
						train_validate_increase_count += 1
						if train_validate_increase_count >= OVERFIT_INCREASE_MAX_ITERATIONS:
							log("(Validation Error - Training Error) Increased too many times (%i)"%OVERFIT_INCREASE_MAX_ITERATIONS)
							bar.finish()
							print("The Validation Error is Diverging from the Training Error.")
							break
					else:
						train_validate_increase_count = 0

					last_training_validation_difference = train_val_diff

			# This if statement handles the logic that ensures that the training
			# stops when the error reaches an apparent minimum.
			if current_iteration > 0:
				error_delta      = np.abs(last_loss - loss_values[current_iteration - 1])
				inside_threshold = error_delta <= FLAT_ERROR_STOP

				if inside_threshold:
					log("Iteration: %i, Error Delta = %f"%(current_iteration, error_delta))

					error_below_threshold_count += 1
					if error_below_threshold_count >= FLAT_ERROR_ITERATIONS:
						if plateau_annealing_count < PLATEAU_ANNEALING_MAX_ITERATIONS:
							# We can partially randomize the network at least one more
							# time.
							# Backup the network before the randomization.
							name = NETWORK_BACKUP_DIR + 'nn_bk_anneal_%i.dat'%(plateau_annealing_count + 1)
							neural_network_data.layers = torch_net.get_network_values()
							neural_network_data.writeNetwork(name)

							log("Network reached local minima, partially randomizing.")
							torch_net, optimizer = partial_randomize(torch_net)
							plateau_annealing_iterations.append(current_iteration)
							plateau_annealing_count += 1
						else:
							log("Error Delta was less than %f for %i iterations."%(FLAT_ERROR_STOP, FLAT_ERROR_ITERATIONS))
							log("Maximum number of randomizations (%i) was also exceeded."%(PLATEAU_ANNEALING_MAX_ITERATIONS))
							bar.finish()
							print("Error Has Reached an Apparent Minimum.")
							break
				else:
					error_below_threshold_count = 0

			if current_iteration % E_VS_V_INTERVAL == 0:
				# Add the current E vs. V values into the log file.
				log_energy_vs_volume()

			

			if current_iteration % NETWORK_BACKUP_INTERVAL == 0:
				# Figure out what to name the file.
				backup_idx = current_iteration // NETWORK_BACKUP_INTERVAL

				if KEEP_BACKUP_HISTORY:
					name       = NETWORK_BACKUP_DIR + 'nn_bk_%i.dat'%backup_idx
				else:
					name       = NETWORK_BACKUP_DIR + 'nn_bk.dat'

				neural_network_data.layers = torch_net.get_network_values()
				neural_network_data.writeNetwork(name)

			# The remaining lines in this loop are where the actaul training takes place.
			if OPTIMIZATION_ALGORITHM == 'LBFGS':
				optimizer.step(closure)
			else:
				optimizer.zero_grad()
				loss = get_loss()
				loss.backward()
				optimizer.step()

			current_iteration += 1

		except KeyboardInterrupt as kbi:
			# The user is trying to terminate the program.
			# Nothing that takes place after the training loop
			# should take very long. Set a flag that tells
			# the script to terminate after dumping all log info.
			terminate_early = True
			log("User input terminated training process early.")
			break


	bar.finish()

	if terminate_early:
		print("Detected Keyboard Interrupt")
		print("Cleaning up and terminating . . .")

	end_time = time()

	if OBJECTIVE_FUNCTION == 'group-targets':

		log("Exporting error as a function of iteration for each group.")
		f = open(GROUP_ERROR_FILE, 'w', FILE_BUFFERING)
		f.write('iteration %s\n'%(' '.join(ordered_group_names)))
		for idx in range(MAXIMUM_TRAINING_ITERATIONS // GROUP_ERROR_RECORD_INTERVAL):
			if group_loss_values[idx] != None:
				f.write('%i %s\n'%(idx * GROUP_ERROR_RECORD_INTERVAL, ' '.join(['%.6E'%e for e in group_loss_values[idx]])))
		f.close()

		# Here we log the training error and validation
		# error on a per-group basis.
		log("Validation and Training Error Per Group")
		log_indent()

		with torch.no_grad():
			# First get the per group training error.
			temp_net          = copy.deepcopy(torch_net)

			calculated_values = temp_net(structure_params)
			difference        = torch.mul(calculated_values - energies, inverse_n_atoms)
			# Multiple by the group reduction matrix in order to get the 
			# error in each group.

			# This multiplication sums the difference squared on a per-group basis.
			per_group_diff_squared_sum = group_reduction_matrix.mm(difference**2)
			per_group_diff_squared_avg = per_group_diff_squared_sum / group_sizes
			per_group_rmse             = torch.sqrt(per_group_diff_squared_avg).cpu()


			# Now get the per group validation error.
			if TRAIN_TO_TOTAL_RATIO != 1.0:
				temp_net.set_reduction_matrix(val_reduction_matrix)
				temp_net = temp_net.cpu()
				calculated_values = temp_net(val_structure_params)
				difference        = torch.mul(calculated_values - val_energies, val_inverse_n_atoms)
				# Multiple by the group reduction matrix in order to get the 
				# error in each group.

				# This multiplication sums the difference squared on a per-group basis.
				per_group_diff_squared_sum = val_group_reduction_matrix.mm(difference**2)
				per_group_diff_squared_avg = per_group_diff_squared_sum / val_group_sizes
				val_per_group_rmse         = torch.sqrt(per_group_diff_squared_avg)

			log("Training: ")
			log_indent()
			for idx in range(len(per_group_rmse)):
				log('%15s = %.3E'%(ordered_group_names[idx], per_group_rmse[idx]))

			log_unindent()

			if TRAIN_TO_TOTAL_RATIO != 1.0:
				log("Validation: ")
				log_indent()
				for idx in range(len(val_per_group_rmse)):
					log('%15s = %.3E'%(val_ordered_group_names[idx], val_per_group_rmse[idx]))

				log_unindent()





		log_unindent()

	if OBJECTIVE_FUNCTION == 'group-targets':
		log("Note: The current objective mode is \"group-targets\", so the following")
		log("      error values are relative to the specified desired error. Depending")
		log("      on your choice of target errors for groups, these error values may")
		log("      be very misleading.")
	log("Final Training Error:   %.3E"%(last_loss))
	if TRAIN_TO_TOTAL_RATIO != 1.0 and len(validation_loss_values) != 0:
		log("Final Validation Error: %.3E"%(validation_loss_values[-1]))


	print("Training Finished")
	print("Time Elapsed: %.1fs"%(end_time - start_time))
	log("Training Finished")
	log("Time Elapsed: %.1fs"%(end_time - start_time))

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

	log("Writing Training Loss File")

	loss_file = open(LOSS_LOG_PATH, 'w', FILE_BUFFERING)
	for loss in loss_values:
		if loss == None:
			break
		loss_file.write('%10.10E\n'%(loss))
	loss_file.close()

	log("Writing Validation Loss File")

	valid_loss_file = open(VALIDATION_LOG_PATH, 'w', FILE_BUFFERING)
	for validation_loss in validation_loss_values:
		valid_loss_file.write('%10.10E\n'%(validation_loss))
	valid_loss_file.close()

	log_unindent()

	if terminate_early:
		exit()

	return plateau_annealing_iterations

def SetupOutputPath(output_path):
	if output_path[-1] != '/':
		output_path += '/'

	global LOG_PATH
	global LSPARAM_FILE
	global NEIGHBOR_FILE
	global NETWORK_BACKUP_DIR
	global LOSS_LOG_PATH
	global VALIDATION_LOG_PATH
	global NEURAL_NETWORK_SAVE_FILE
	global E_VS_V_FILE
	global GROUP_ERROR_FILE

	LOG_PATH                 = output_path + LOG_PATH
	LSPARAM_FILE             = output_path + LSPARAM_FILE
	NEIGHBOR_FILE            = output_path + NEIGHBOR_FILE
	NETWORK_BACKUP_DIR       = output_path + NETWORK_BACKUP_DIR
	LOSS_LOG_PATH            = output_path + LOSS_LOG_PATH
	VALIDATION_LOG_PATH      = output_path + VALIDATION_LOG_PATH
	NEURAL_NETWORK_SAVE_FILE = output_path + NEURAL_NETWORK_SAVE_FILE
	E_VS_V_FILE              = output_path + E_VS_V_FILE
	GROUP_ERROR_FILE         = output_path + GROUP_ERROR_FILE			

	paths = [
		LOG_PATH,
		LSPARAM_FILE,
		NEIGHBOR_FILE,
		NETWORK_BACKUP_DIR,
		LOSS_LOG_PATH,
		VALIDATION_LOG_PATH,
		NEURAL_NETWORK_SAVE_FILE,
		E_VS_V_FILE,
		GROUP_ERROR_FILE
	]
	
	for path in paths:
		# We need to split the path up on '/' characters
		# and ensure that each subdirectory exists.

		current = ''
		if path[-1] == '/':
			spl = path.split('/')
		else:
			spl = path.split('/')[:-1]

		for p in spl:
			current += p

			if current != '':
				if not os.path.exists(current):
					os.mkdir(current)

			current += '/'


if __name__ == '__main__':

	args = []
	for s in sys.argv[1:]:
		if s[0] != '-':
			args.append(s)
		else:
			args.append(s.lower())

	output_path = None
	if '--directory' in args:
		output_path = str(args[args.index('--directory') + 1])
	if '-d' in args:
		output_path = str(args[args.index('-d') + 1])

	if output_path != None:
		SetupOutputPath(output_path)

		Util.init(LOG_PATH)
		log('---------- IMPORTANT ----------')
		log('Output was redirected to: %s\n'%output_path)
	else:
		Util.init()
	
	log("---------- Program Started ----------\n")

	log("Command Line: %s\n"%' '.join(sys.argv))

	program_start = time()

	# This will take the contents of the configuration file and
	# print them in a nice format in the log file. This process
	# does not keep comme.confignts that are in the configuration file.
	Util.LogConfiguration()

	# ----------------------------------------
	# Command Line Argument Processing
	# ----------------------------------------

	

	compute_gis  = False # Whether or not to compute structure params from poscar data.
	run_training = False # Whether or not to traing the neural network against structure params.
	graph_error  = False # Whether or not to produce a plot of error as a function of 
	                     # the training iteration at the end of the training process.
	graph_val    = False # Whether or not to plot validation error as a function of iteration
	                     # at the end of the process.
	graph_group  = False # Whether or not to graph the per-group error at the end.
	force_cpu    = False # Whether or not to keep all execution on cpu
	randomize_nn = False # Whether or not to ignore configuration and randomize the nn.
	unsupervised = False # Whether or not to use minimal progress bars.

	# Here we expand single hyphen arguments.
	# ex: -gte => -g -t -e

	tmp = []
	for arg in args:
		if len(arg) > 1 and arg[:2] != '--' and arg[0] == '-' and len(arg) > 2:
			# This is a single hypen argument specifying
			# more than one option.
			tmp.extend(['-' + c for c in arg[1:]])
		else:
			tmp.append(arg)
	args = tmp


	group_to_highlight = None

	if len(args) == 0:
		run_training = True
	else:


		if '--help' in args or '-h' in args:
			print(help_str)
			exit()
		if '--compare' in args or '-c' in args:
			if len(args) != 3:
				print("Invalid Argument Combination:")
				print("If -c or --compare is specified, you must specify two file names after and nothing else.")
				exit()

			first_file  = args[1]
			second_file = args[2]
			CompareStructureParameters(first_file, second_file)
			exit()

		if '--highlight' in args:
			group_to_highlight = args[args.index('--highlight') + 1]
		if '-i' in args:
			group_to_highlight = args[args.index('-i') + 1]

		if '--target-errors' in args:
			if OBJECTIVE_FUNCTION != 'group-targets':
				print("OBJECTIVE_FUNCTION != 'group-targets'")
				print("You might want to change that.")

			start_idx = args.index('--target-errors') + 1

			global DEFAULT_TARGET
			DEFAULT_TARGET = float(args[start_idx])

			global SUBGROUP_TARGETS
			for i in range(start_idx + 1, len(args)):
				arg = args[i]
				name, target = arg.split(':')
				SUBGROUP_TARGETS[name] = float(target)

		gpu_affinity = 0
		if '--gpu-affinity' in args:
			gpu_affinity = int(args[args.index('--gpu-affinity') + 1])
			log("GPU Affinity: %i"%gpu_affinity)
		if '--n-threads' in args:
			os.environ["OMP_NUM_THREADS"] = str(args[args.index('--n-threads') + 1])
			os.environ["MKL_NUM_THREADS"] = str(args[args.index('--n-threads') + 1])
			torch.set_num_threads(int(args[args.index('--n-threads') + 1]))
		if '-n' in args:
			os.environ["OMP_NUM_THREADS"] = str(args[args.index('-n') + 1])
			os.environ["MKL_NUM_THREADS"] = str(args[args.index('-n') + 1])
			torch.set_num_threads(int(args[args.index('-n') + 1]))

		if '--unsupervised' in args or '-u' in args:
			unsupervised = True
			Util.set_mode(unsupervised)
		if '--compute-gis' in args or '-g' in args:
			compute_gis = True
		if '--randomize-nn' in args or '-r' in args:
			randomize_nn = True
		if '--run-training' in args or '-t' in args:
			run_training = True
		if '--graph-error' in args or '-e' in args:
			graph_error = True
		if '--graph-val' in args or '-v' in args:
			graph_val = True
		if '--group-error' in args or '-s' in args:
			graph_group = True
		if '--cpu' in args or '-p' in args:
			force_cpu = True

	# By this point we know what operations the user requested.
	# Start running them, in the logical order.

	plateau_annealing_iterations = None

	if compute_gis:
		ComputeStructureParameters()

	if run_training:
		plateau_annealing_iterations = TrainNetwork(force_cpu, randomize_nn, gpu_affinity)

	if graph_error:
		import matplotlib.pyplot as plt
		GraphError(graph_error, graph_val, plateau_annealing_iterations)

	if graph_group:
		import matplotlib.pyplot as plt
		GraphGroupError(group_to_highlight)


	program_end = time()
	log("Total Run Time: %.1fs"%(program_end - program_start))