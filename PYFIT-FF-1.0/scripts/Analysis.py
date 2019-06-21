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
	no_comparison_plots            = '--no-comparison-plots'            in args # Don't show correlation - rmse plots
	no_histograms                  = '--no-histograms'                  in args # Don't show histograms
	small_mode                     = '--small-mode'                     in args

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
		if small_mode:
			contents_dir = new_dir
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
		else:
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


	composite_sort = [(a, b, c) for a, b, c in zip(results, data, locations)]
	composite_sort = sorted(composite_sort, key=lambda x: x[2])

	results   = [i[0] for i in composite_sort]
	data      = [i[1] for i in composite_sort]
	locations = [i[2] for i in composite_sort]

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

		all_00 = []
		for result, d in zip(results, data):
			mode = result['parameter_set']['gi_mode']
			act  = result['parameter_set']['activation_function']
			if mode == 0 and act == 0:
				all_00.append(d)

		all_01 = []
		for result, d in zip(results, data):
			mode = result['parameter_set']['gi_mode']
			act  = result['parameter_set']['activation_function']
			if mode == 0 and act == 1:
				all_01.append(d)

		all_10 = []
		for result, d in zip(results, data):
			mode = result['parameter_set']['gi_mode']
			act  = result['parameter_set']['activation_function']
			if mode == 1 and act == 0:
				all_10.append(d)

		all_11 = []
		for result, d in zip(results, data):
			mode = result['parameter_set']['gi_mode']
			act  = result['parameter_set']['activation_function']
			if mode == 1 and act == 1:
				all_11.append(d)

		network_properties_00 = []
		network_properties_01 = []
		network_properties_10 = []
		network_properties_11 = []

		for group_data in all_00:
			group = group_data['feature-output-correlations']
			for network in group:
				net              = group[network]
				parameter_values = []
				for parameter in net['final']:
					parameter_values.append(parameter['coefficient'])
				network_properties_00.append(parameter_values)

		for group_data in all_01:
			group = group_data['feature-output-correlations']
			for network in group:
				net              = group[network]
				parameter_values = []
				for parameter in net['final']:
					parameter_values.append(parameter['coefficient'])
				network_properties_01.append(parameter_values)

		for group_data in all_10:
			group = group_data['feature-output-correlations']
			for network in group:
				net              = group[network]
				parameter_values = []
				for parameter in net['final']:
					parameter_values.append(parameter['coefficient'])
				network_properties_10.append(parameter_values)

		for group_data in all_11:
			group = group_data['feature-output-correlations']
			for network in group:
				net              = group[network]
				parameter_values = []
				for parameter in net['final']:
					parameter_values.append(parameter['coefficient'])
				network_properties_11.append(parameter_values)


		# Now that we have the parameters for each network, we essentially transpose
		# that matrix so that the first index is the parameter and the second is the
		# network, instead of vice-verse.
		color  = ['red', 'green', 'blue', 'orange']
		offset = [0, 36, 72, 108]
		name   = ['Mode 0 0', 'Mode 0 1', 'Mode 1 0', 'Mode 1 1']
		fig, ax = plt.subplots(1, 1)
		for idx, chunk in enumerate([network_properties_00, network_properties_01, network_properties_10, network_properties_11]):
			coefficient_values = np.array(chunk).transpose()
			n_networks_total   = coefficient_values.shape[1]
			n_params_total     = coefficient_values.shape[0]

			# Now we plot all of the data in one graph.
			

			for parameter in range(n_params_total):
				rng = np.array(range(n_networks_total)) + offset[idx]
				ax.plot(rng, coefficient_values[parameter], linewidth=1, linestyle=':', color=color[idx])

		ax.set_xlabel("Network")
		ax.set_ylabel("Coefficient")
		ax.set_xticks([18, 54, 90, 126])
		ax.set_xticklabels(name)
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


	# ============================================================
	# ERROR PLOTS
	# ============================================================

	if not no_comparison_plots:

		figures_of_merit = []
		ff_correlation   = []
		fc_correlation   = []
		mean_rmse        = []

		for idx, info in enumerate(results):
			rmse = info['scores']['mean_rmse']
			mean_rmse.append(rmse)
			figures_of_merit.append(info['scores']['figure_of_merit'])
			ff_correlation.append(info['scores']['mean_ff_correlation'])
			fc_correlation.append(info['scores']['mean_fc_correlation'])

		fig, axes = plt.subplots(2, 2)

		# ------------------------------------------------------------
		# Error vs. Figure of Merit
		# ------------------------------------------------------------

		mean_ = axes[0, 0].scatter(figures_of_merit, mean_rmse, s = 6)

		y_max = max([max(mean_rmse), 0.2])

		x_major_ticks = np.linspace(0, 1.0, 11)
		x_minor_ticks = np.linspace(0, 1.0, 51)
		y_major_ticks = np.linspace(0, y_max, 6)
		y_minor_ticks = np.linspace(0, y_max, 26)

		axes[0, 0].set_xticks(x_major_ticks)
		axes[0, 0].set_xticks(x_minor_ticks, minor=True)
		axes[0, 0].set_yticks(y_major_ticks)
		axes[0, 0].set_yticks(y_minor_ticks, minor=True)

		# And a corresponding grid
		axes[0, 0].grid(which='both')

		# Or if you want different settings for the grids:
		axes[0, 0].grid(which='minor', alpha=0.2)
		axes[0, 0].grid(which='major', alpha=0.5)
		axes[0, 0].set_xlabel("Figure of Merit")
		axes[0, 0].set_ylabel("RMSE")
		axes[0, 0].set_xlim(0.0, 1.0)
		axes[0, 0].set_ylim(0.0, y_max)
		axes[0, 0].set_title("Error vs. Figure of Merit")


		# ------------------------------------------------------------
		# Error vs. Mean Feature Feature Correlation
		# ------------------------------------------------------------

		ff_mean_ = axes[0, 1].scatter(ff_correlation, mean_rmse, s = 6)

		y_max = max([max(mean_rmse), 0.2])

		x_major_ticks = np.linspace(0, 1.0, 11)
		x_minor_ticks = np.linspace(0, 1.0, 51)
		y_major_ticks = np.linspace(0, y_max, 6)
		y_minor_ticks = np.linspace(0, y_max, 26)

		axes[0, 1].set_xticks(x_major_ticks)
		axes[0, 1].set_xticks(x_minor_ticks, minor=True)
		axes[0, 1].set_yticks(y_major_ticks)
		axes[0, 1].set_yticks(y_minor_ticks, minor=True)

		# And a corresponding grid
		axes[0, 1].grid(which='both')

		# Or if you want different settings for the grids:
		axes[0, 1].grid(which='minor', alpha=0.2)
		axes[0, 1].grid(which='major', alpha=0.5)

		axes[0, 1].set_xlabel("Mean Feature Feature Correlation")
		axes[0, 1].set_ylabel("RMSE")
		axes[0, 1].set_xlim(0.0, 1.0)
		axes[0, 1].set_ylim(0.0, y_max)
		axes[0, 1].set_title("Error vs. Mean Feature Feature Correlation")

		# ------------------------------------------------------------
		# Error vs. Mean Feature Classification Correlation
		# ------------------------------------------------------------

		fc_mean_ = axes[1, 0].scatter(fc_correlation, mean_rmse, s = 6)

		y_max = max([max(mean_rmse), 0.2])

		x_major_ticks = np.linspace(0, 1.0, 11)
		x_minor_ticks = np.linspace(0, 1.0, 51)
		y_major_ticks = np.linspace(0, y_max, 6)
		y_minor_ticks = np.linspace(0, y_max, 26)

		axes[1, 0].set_xticks(x_major_ticks)
		axes[1, 0].set_xticks(x_minor_ticks, minor=True)
		axes[1, 0].set_yticks(y_major_ticks)
		axes[1, 0].set_yticks(y_minor_ticks, minor=True)

		# And a corresponding grid
		axes[1, 0].grid(which='both')

		# Or if you want different settings for the grids:
		axes[1, 0].grid(which='minor', alpha=0.2)
		axes[1, 0].grid(which='major', alpha=0.5)

		axes[1, 0].set_xlabel("Mean Feature Classification Correlation")
		axes[1, 0].set_ylabel("RMSE")
		axes[1, 0].set_xlim(0.0, 1.0)
		axes[1, 0].set_ylim(0.0, y_max)
		axes[1, 0].set_title("Error vs. Mean Feature Classification Correlation")

		# ------------------------------------------------------------
		# Figure of Merit vs. Hyperparameter Set
		# ------------------------------------------------------------

		mh = axes[1, 1].scatter(range(len(locations)), figures_of_merit, s = 6)
		axes[1, 1].set_xlabel("Hyperparameter Set Index")
		axes[1, 1].set_ylabel("Figure of Merit")
		axes[1, 1].set_title("Figure of Merit vs. Hyperparameter Set Index")

		def format_display(**kwargs):
			print(locations[kwargs['ind'][0]])
			return None

		datacursor(formatter=format_display)

		plt.show()


	# ============================================================
	# PERCENTILE HISTOGRAMS
	# ============================================================

	if not no_histograms:

		properties_for_sorting = []

		for idx, info in enumerate(results):
			props = {}

			props['location']        = locations[idx]
			props['figure_of_merit'] = info['scores']['figure_of_merit']
			props['n_r0']            = len(info['parameter_set']['r_0_values'])
			props['n_l']             = len(info['parameter_set']['legendre_polynomials'])
			props['mode']            = info['parameter_set']['gi_mode']
			props['shift']           = info['parameter_set']['gi_shift']
			props['activation']      = info['parameter_set']['activation_function']
			props['sigma']           = info['parameter_set']['gi_sigma']
			props['rmse']            = info['scores']['mean_rmse']
			properties_for_sorting.append(props)

		sorted_by_fm   = sorted(properties_for_sorting, key=lambda x: x['figure_of_merit'])
		sorted_by_rmse = sorted(properties_for_sorting, key=lambda x: -x['rmse'])

		def gen_bar(vals):
			locations = np.unique(vals).tolist()
			vals      = np.array(vals)
			counts    = []

			for location in locations:
				counts.append(len(vals[(vals == location)]))

			return locations, counts

		def show_histograms(dataset, name):

			fig, axes = plt.subplots(2, 3)

			loc, count = gen_bar([p['n_r0'] for p in dataset])
			axes[0, 0].bar(loc, count, edgecolor='black', linewidth=0.5)
			axes[0, 0].set_xlabel("# of $r_0$ Values")
			axes[0, 0].set_ylabel("Quantity")
			axes[0, 0].set_xticks(loc)
			axes[0, 0].set_xticklabels(loc)
			axes[0, 0].set_title("$r_0$ (%s)"%name)

			loc, count = gen_bar([p['n_l'] for p in dataset])
			axes[0, 1].bar(loc, count, edgecolor='black', linewidth=0.5)
			axes[0, 1].set_xlabel("# of Legendre Polynomials")
			axes[0, 1].set_ylabel("Quantity")
			axes[0, 1].set_xticks(loc)
			axes[0, 1].set_xticklabels(loc)
			axes[0, 1].set_title("$P_l$ (%s)"%name)

			loc, count = gen_bar([p['mode'] for p in dataset])
			axes[0, 2].bar(loc, count, edgecolor='black', linewidth=0.5)
			axes[0, 2].set_xlabel("Gi Mode")
			axes[0, 2].set_ylabel("Quantity")
			axes[0, 2].set_xticks(loc)
			axes[0, 2].set_xticklabels(['Normal', 'Log'])
			axes[0, 2].set_title("Mode (%s)"%name)

			loc, count = gen_bar([p['shift'] for p in dataset])
			axes[1, 0].bar(loc, count, edgecolor='black', linewidth=0.5, width=0.2)
			axes[1, 0].set_xlabel("Gi Shift")
			axes[1, 0].set_ylabel("Quantity")
			axes[1, 0].set_xticks(loc)
			axes[1, 0].set_xticklabels(loc)
			axes[1, 0].set_title("Shift (%s)"%name)

			loc, count = gen_bar([p['activation'] for p in dataset])
			axes[1, 1].bar(loc, count, edgecolor='black', linewidth=0.5)
			axes[1, 1].set_xlabel("Activation Function")
			axes[1, 1].set_ylabel("Quantity")
			axes[1, 1].set_xticks(loc)
			axes[1, 1].set_xticklabels(['Sigmoid', 'Shifted Sigmoid'])
			axes[1, 1].set_title("Activation Function (%s)"%name)

			loc, count = gen_bar([p['sigma'] for p in dataset])
			axes[1, 2].bar(loc, count, edgecolor='black', linewidth=0.5, width=0.12)
			axes[1, 2].set_xlabel("$\\sigma$")
			axes[1, 2].set_ylabel("Quantity")
			axes[1, 2].set_xticks(loc)
			axes[1, 2].set_xticklabels(loc)
			axes[1, 2].set_title("Sigma (%s)"%name)

			plt.show()

		

		_10_percent_n         = int(round(0.10 * len(sorted_by_fm)))
		fm_top_10_percent    = sorted_by_fm[-_10_percent_n:]
		fm_bottom_10_percent = sorted_by_fm[:_10_percent_n]

		show_histograms(fm_top_10_percent, "Top 10 Percent by Figure of Merit")
		show_histograms(fm_bottom_10_percent, "Bottom 10 Percent by Figure of Merit")
	