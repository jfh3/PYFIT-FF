import matplotlib.pyplot as plt
import numpy             as np
import sys
import json
import os
from   mpldatacursor import datacursor


import warnings
warnings.filterwarnings("ignore")

def print_np_as_grid(arr, tabs=0):
	np.set_printoptions(precision=3)
	s = str(arr)
	s = s.replace('[', ' ').replace(']', '')

	if tabs != 0:
		lines = s.split('\n')
		lines = ['\t'*tabs + line for line in lines]
		s = '\n'.join(lines)

	print(s)

if __name__ == '__main__':
	root_dir = sys.argv[1] # The directory that contains all of the idx_* directories

	args = [arg.lower() for arg in sys.argv[2:]]

	# These two options pretty much only pertain to the runs that were meant to 
	# establish that for the same Gi's, the networks will all use them the same
	# way even if their architecture is different.
	ignore_non_convergence         = '--ignore-non-convergence'         in args # Analyze non-converged data anyways
	find_convergence_point         = '--find-convergence-point'         in args
	plot_cross_network_convergence = '--plot-cross-network-convergence' in args

	print("Analyzing %s"%root_dir)

	if ignore_non_convergence:
		print("Analyzing Non-Converged Data")


	# ==================================================
	# Data Loading
	# ==================================================

	results   = [] # master_results.json file data
	data      = [] # final_data.json file data
	locations = [] # directory the data is in

	divergent_results   = []
	divergent_locations = []

	for subdir in os.listdir(root_dir):
		new_dir = root_dir + subdir + '/'
		for subsubdir in os.listdir(new_dir):
			contents_dir = new_dir + subsubdir + '/'
			results_file = contents_dir + 'master_results.json'
			data_file    = contents_dir + 'final_data.json'
			if os.path.isfile(results_file):

				f = open(results_file, 'r')
				raw = f.read()
				f.close()

				f = open(data_file, 'r')
				raw2 = f.read()
				f.close()

				if 'NaN' in raw or 'Infinity' in raw:
					to_parse = raw.replace('NaN', '0.0').replace('Infinity', '0.0')
					divergent_results.append(json.loads(to_parse))
					divergent_locations.append(results_file)
				else:
					results.append(json.loads(raw))
					locations.append(contents_dir)
					data.append(json.loads(raw2))


	# ==================================================
	# Data Sorting and Filtering
	# ==================================================

	# Figure out how many had non-parameter convergence.

	non_param_convergence           = []
	non_param_convergence_data      = []
	non_param_convergence_locations = []

	tmp_result   = []
	tmp_data     = []
	tmp_location = []

	for result, data, location in zip(results, data, locations):
		if False in result['all_params_converged']:
			non_param_convergence.append(result)
			non_param_convergence_data.append(data)
			non_param_convergence_locations.append(location)
		else:
			tmp_result.append(result)
			tmp_data.append(data)
			tmp_location.append(location)

	results   = tmp_result
	data      = tmp_data
	locations = tmp_location


	# Figure out how many had non-convergence between networks.

	non_net_convergence           = []
	non_net_convergence_data      = []
	non_net_convergence_locations = []

	tmp_result   = []
	tmp_data     = []
	tmp_location = []

	for result, data, location in zip(results, data, locations):
		if not result['all_params_converged_between_networks']:
			non_net_convergence.append(result)
			non_net_convergence_data.append(data)
			non_net_convergence_locations.append(location)
		else:
			tmp_result.append(result)
			tmp_data.append(data)
			tmp_location.append(location)

	results   = tmp_result
	data      = tmp_data
	locations = tmp_location

	# Summarize what was discovered.
	print("General Results:")
	print("\tGood Results                 = %i"%(len(results)))
	print("\tDivergent Networks           = %i"%(len(divergent_results)))
	print("\tNon Converged Networks       = %i"%(len(non_param_convergence)))
	print("\tNon Converged Network Groups = %i"%(len(non_net_convergence)))

	# The the user says so, analyze the non-converged data anyways.
	if ignore_non_convergence:
		results.extend(non_param_convergence)
		results.extend(non_net_convergence)

		data.extend(non_param_convergence_data)
		data.extend(non_net_convergence_data)

		locations.extend(non_param_convergence_locations)
		locations.extend(non_net_convergence_locations)


	# ==================================================
	# Convergence Analysis
	# ==================================================

	if plot_cross_network_convergence:
		# We need to get the final value of each parameter for all networks.
		# We will then plot the network on the x-axis and its final value for
		# the property on the y-axis. We will have one series in this plot for
		# each property.

		# --------------------------------------------------
		# This is where a filter should be added if you are 
		# preparing a plot for the final paper.
		# --------------------------------------------------

		data_to_use = []
		for result, d in zip(results, data):
			mode = result['parameter_set']['gi_mode']
			act  = result['parameter_set']['activation_function']
			if mode == 1 and act == 1:
				data_to_use.append(d)

		network_properties = []

		for group_data in data_to_use:
			group = group_data['feature-output-correlations']
			for network in group:
				net              = group[network]
				parameter_values = []
				for parameter in net['final']:
					parameter_values.append(parameter['coefficient'])
				network_properties.append(parameter_values)


		# Now that we have the parameters for each network, we essentially transpose
		# that matrix so that the first index is the parameter and the second is the
		# network, instead of vice-verse.
		coefficient_values = np.array(network_properties).transpose()
		n_networks_total   = coefficient_values.shape[1]
		n_params_total     = coefficient_values.shape[0]

		# Now we plot all of the data in one graph.
		fig, ax = plt.subplots(1, 1)

		for parameter in range(n_params_total):
			ax.plot(range(n_networks_total), coefficient_values[parameter], linewidth=1, linestyle=':')

		ax.set_xlabel("Network")
		ax.set_ylabel("Coefficient")
		ax.set_title(
			"Final Value of Correlation Between Feature and Output for All Networks and All Features"
		)
		plt.show()

		# Determine and print the standard deviation across all networks, on a per parameter basis.
		standard_deviations = []
		for parameter in range(n_params_total):
			standard_deviations.append(coefficient_values[parameter].std())

		standard_deviations = np.array(standard_deviations)
		
		print("Mean Standard Deviation = %1.2f"%standard_deviations.mean())
		print("Min  Standard Deviation = %1.2f"%standard_deviations.min())
		print("Max  Standard Deviation = %1.2f"%standard_deviations.max())
		print_np_as_grid(standard_deviations, 1)

	if find_convergence_point:
		# We need the feature output correlation for all features, for all networks
		# and at all backup stages. 

		network_convergences = []

		for group_data in data:
			group = group_data['feature-output-correlations']
			for network in group:
				net              = group[network]
				stages           = []

				# We want to form a list of backups and sort them so that they
				# are in order.

				for backup in net['backups']:
					stages.append((backup, net['backups'][backup]))


				# Sort them by name.
				stages = [stg[1] for stg in sorted(stages, key=lambda x: x[0])]
				stages.append(net['final'])

				# Now we have a list of backups in order, each containing the correlation
				# coefficient for every parameter. Now we turn this into a list where the
				# first axis is the parameter, and the second axis is the stage in the 
				# training process, from start to finish.
				param_values = []
				for stage in stages:
					param_values.append([coeff['coefficient'] for coeff in stage])

				param_values = np.array(param_values).transpose()
				
				# We now have a list of parameter coefficient with respect to training iteration
				# for all parameters of this neural network.
				# Add it to the list.

				network_convergences.append(param_values)


		# Now that we have all of the data, take each coefficient for each network and
		# move a sliding window within which the standard deviation is calculated backwards
		# in order to find the earliest point at which all networks and all properties
		# became converged during this process. This value will largely dictate how many
		# training iterations need to be used during the large sweep.

		window_size   = 8
		threshold_std = 0.02

		highest_divergence_point = 0

		for network in network_convergences:
			for coefficient in range(network.shape[0]):
				# Slide the window back until the standard deviation 
				# within it exceeds a threshold value.

				# Start at the end.
				window_end   = network.shape[1]
				window_start = window_end - window_size

				divergence_point = 0

				while window_start >= 0 and window_start >= highest_divergence_point:
					window_std = network[coefficient][window_start:window_end].std()
					if window_std >= threshold_std:
						divergence_point = window_start
						break

					window_end   -= 1
					window_start = window_end - window_size

				# We have the divergence point for this coefficient, of this network.
				if divergence_point >= highest_divergence_point:
					highest_divergence_point = divergence_point

		print("The highest point at which a network started to converge was: %i"%highest_divergence_point)