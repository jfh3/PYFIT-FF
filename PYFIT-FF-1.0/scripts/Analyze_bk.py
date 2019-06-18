import matplotlib.pyplot as plt
import numpy             as np
import sys
import json
import os
from   mpldatacursor import datacursor


import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
	# We want to load all of the master_results.json files into memory.
	# Some of them will inevitably be invalid.

	results_data = []
	locations    = []

	results_dir = 'feature-set-02/'
	for subdir in os.listdir(results_dir):
		new_dir = results_dir + subdir + '/'
		for subsubdir in os.listdir(new_dir):
			contents_dir = new_dir + subsubdir + '/'
			results_file = contents_dir + 'master_results.json'
			if os.path.isfile(results_file):
				f = open(results_file, 'r')
				raw = f.read()
				f.close()
				to_parse = raw.replace('NaN', '0.0').replace('Infinity', '0.0')
				locations.append(results_file)
				results_data.append(json.loads(to_parse))

	tmp_sort = [(results_data[i], locations[i]) for i in range(len(results_data))]
	tmp_sort = sorted(tmp_sort, key=lambda x: int(x[1].split('_')[1].split('/')[0]))
	results_data = [i[0] for i in tmp_sort]
	locations    = [i[1] for i in tmp_sort]


	n_broken = 0
	for data in results_data:
		if data['scores']['mean_rmse'] == 0.0:
			n_broken += 1

	print("analyzing %i feature sets"%len(results_data))
	print('%i broken'%n_broken)

	locations_clean  = []
	figures_of_merit = []
	ff_correlation   = []
	fc_correlation   = []
	n_r0             = []
	mean_rmse        = []
	min_rmse         = []
	max_rmse         = []

	for idx, data in enumerate(results_data):
		rmse = data['scores']['mean_rmse']
		_min = data['scores']['min_rmse']
		_max = data['scores']['max_rmse']

		if rmse < 1e-3 and rmse != 0.0:
			print('Good Result: %s'%(locations[idx]))
		elif rmse != 0.0:
			locations_clean.append(locations[idx])
			figures_of_merit.append(data['scores']['figure_of_merit'])
			mean_rmse.append(rmse)
			min_rmse.append(_min)
			max_rmse.append(_max)
			ff_correlation.append(data['scores']['mean_ff_correlation'])
			fc_correlation.append(data['scores']['mean_fc_correlation'])
			n_r0.append(len(data['parameter_set']['r_0_values']))

	print("Lowest Average RMSE: %f"%min(mean_rmse))
	print("Lowest Min     RMSE: %f"%min(min_rmse))

	# ============================================================
	# ERROR PLOTS
	# ============================================================

	fig, axes = plt.subplots(2, 2)

	# ------------------------------------------------------------
	# Error vs. Figure of Merit
	# ------------------------------------------------------------

	mean_ = axes[0, 0].scatter(figures_of_merit, mean_rmse, s = 6)
	min_  = axes[0, 0].scatter(figures_of_merit, min_rmse,  s = 6)
	max_  = axes[0, 0].scatter(figures_of_merit, max_rmse,  s = 6)

	y_max = min([max([max(mean_rmse), max(min_rmse), max(max_rmse)]), 0.2])

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

	axes[0, 0].legend([mean_, min_, max_], ["Mean", "Minimum", "Maximum"])
	axes[0, 0].set_xlabel("Figure of Merit")
	axes[0, 0].set_ylabel("RMSE")
	axes[0, 0].set_xlim(0.0, 1.0)
	axes[0, 0].set_ylim(0.0, y_max)
	axes[0, 0].set_title("Error vs. Figure of Merit")


	# ------------------------------------------------------------
	# Error vs. Mean Feature Feature Correlation
	# ------------------------------------------------------------

	ff_mean_ = axes[0, 1].scatter(ff_correlation, mean_rmse, s = 6)
	ff_min_  = axes[0, 1].scatter(ff_correlation, min_rmse,  s = 6)
	ff_max_  = axes[0, 1].scatter(ff_correlation, max_rmse,  s = 6)

	y_max = min([max([max(mean_rmse), max(min_rmse), max(max_rmse)]), 0.2])

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

	axes[0, 1].legend([ff_mean_, ff_min_, ff_max_], ["Mean", "Minimum", "Maximum"])
	axes[0, 1].set_xlabel("Mean Feature Feature Correlation")
	axes[0, 1].set_ylabel("RMSE")
	axes[0, 1].set_xlim(0.0, 1.0)
	axes[0, 1].set_ylim(0.0, y_max)
	axes[0, 1].set_title("Error vs. Mean Feature Feature Correlation")

	# ------------------------------------------------------------
	# Error vs. Mean Feature Classification Correlation
	# ------------------------------------------------------------

	fc_mean_ = axes[1, 0].scatter(fc_correlation, mean_rmse, s = 6)
	fc_min_  = axes[1, 0].scatter(fc_correlation, min_rmse,  s = 6)
	fc_max_  = axes[1, 0].scatter(fc_correlation, max_rmse,  s = 6)

	y_max = min([max([max(mean_rmse), max(min_rmse), max(max_rmse)]), 0.2])

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

	axes[1, 0].legend([fc_mean_, fc_min_, fc_max_], ["Mean", "Minimum", "Maximum"])
	axes[1, 0].set_xlabel("Mean Feature Classification Correlation")
	axes[1, 0].set_ylabel("RMSE")
	axes[1, 0].set_xlim(0.0, 1.0)
	axes[1, 0].set_ylim(0.0, y_max)
	axes[1, 0].set_title("Error vs. Mean Feature Classification Correlation")

	# ------------------------------------------------------------
	# Figure of Merit vs. Hyperparameter Set
	# ------------------------------------------------------------

	mh = axes[1, 1].scatter(range(len(locations_clean)), figures_of_merit, s = 6)
	axes[1, 1].set_xlabel("Hyperparameter Set Index")
	axes[1, 1].set_ylabel("Figure of Merit")
	axes[1, 1].set_title("Figure of Merit vs. Hyperparameter Set Index")

	def format_display(**kwargs):
		print(locations_clean[kwargs['ind'][0]])
		return None

	datacursor(formatter=format_display)

	plt.show()


	# ============================================================
	# PERCENTILE HISTOGRAMS
	# ============================================================

	properties_for_sorting = []

	for idx, data in enumerate(results_data):
		props = {}

		if rmse != 0.0:
			props['location']        = locations[idx]
			props['figure_of_merit'] = data['scores']['figure_of_merit']
			props['n_r0']            = len(data['parameter_set']['r_0_values'])
			props['n_l']             = len(data['parameter_set']['legendre_polynomials'])
			props['mode']            = data['parameter_set']['gi_mode']
			props['shift']           = data['parameter_set']['gi_shift']
			props['activation']      = data['parameter_set']['activation_function']
			props['sigma']           = data['parameter_set']['gi_sigma']
			props['rmse']            = data['scores']['mean_rmse']
			properties_for_sorting.append(props)

	sorted_by_fm = sorted(properties_for_sorting, key=lambda x: x['figure_of_merit'])


	all_r0 = np.unique([p['n_r0'] for p in sorted_by_fm])
	all_l  = np.unique([p['n_l'] for p in sorted_by_fm])
	all_s  = np.unique([p['sigma'] for p in sorted_by_fm])

	def show_histograms(dataset, name):
		fig, axes = plt.subplots(2, 3)

		n_r0 = [p['n_r0'] for p in dataset]
		bins = axes[0, 0].hist(n_r0, bins=len(all_r0), align='left', edgecolor='black', linewidth=0.5)
		axes[0, 0].set_xlabel("# of $r_0$ Values")
		axes[0, 0].set_ylabel("Quantity")
		axes[0, 0].set_xticks(bins[1])
		axes[0, 0].set_xticklabels(all_r0)
		axes[0, 0].set_title("$r_0$ (%s)"%name)

		n_l  = [p['n_l'] for p in dataset]
		bins = axes[0, 1].hist(n_l, bins=len(all_l), align='left', edgecolor='black', linewidth=0.5)
		axes[0, 1].set_xlabel("# of Legendre Polynomials")
		axes[0, 1].set_ylabel("Quantity")
		axes[0, 1].set_xticks(bins[1])
		axes[0, 1].set_xticklabels(all_l)
		axes[0, 1].set_title("$P_l$ (%s)"%name)

		mode = [p['mode'] for p in dataset]
		bins = axes[0, 2].hist(mode, bins=2, align='left', edgecolor='black', linewidth=0.5)
		axes[0, 2].set_xlabel("Gi Mode")
		axes[0, 2].set_ylabel("Quantity")
		axes[0, 2].set_xticks(bins[1])
		axes[0, 2].set_xticklabels(['Normal', 'Log'])
		axes[0, 2].set_title("Mode (%s)"%name)

		shift = [p['shift'] for p in dataset]
		bins = axes[1, 0].hist(shift, bins=2, align='left', edgecolor='black', linewidth=0.5)
		axes[1, 0].set_xlabel("Gi Shift")
		axes[1, 0].set_ylabel("Quantity")
		axes[1, 0].set_xticks(bins[1])
		axes[1, 0].set_xticklabels([0, 0.5])
		axes[1, 0].set_title("Shift (%s)"%name)

		activation = [p['activation'] for p in dataset]
		bins = axes[1, 1].hist(activation, bins=2, align='left', edgecolor='black', linewidth=0.5)
		axes[1, 1].set_xlabel("Activation Function")
		axes[1, 1].set_ylabel("Quantity")
		axes[1, 1].set_xticks(bins[1])
		axes[1, 1].set_xticklabels(['Sigmoid', 'Shifted Sigmoid'])
		axes[1, 1].set_title("Activation Function (%s)"%name)

		sigma = [p['sigma'] for p in dataset]
		bins = axes[1, 2].hist(sigma, bins=len(all_s), align='left', edgecolor='black', linewidth=0.5)
		axes[1, 2].set_xlabel("$\\sigma$")
		axes[1, 2].set_ylabel("Quantity")
		axes[1, 2].set_xticks(bins[1])
		axes[1, 2].set_xticklabels(all_s)
		axes[1, 2].set_title("Sigma (%s)"%name)

		plt.show()

	
	top_2_5_percent_n = int(round(0.025 * len(sorted_by_fm)))
	top_2_5_percent   = sorted_by_fm[-top_2_5_percent_n:]

	top_5_percent_n = int(round(0.05 * len(sorted_by_fm)))
	top_5_percent   = sorted_by_fm[-top_5_percent_n:]

	top_10_percent_n = int(round(0.1 * len(sorted_by_fm)))
	top_10_percent   = sorted_by_fm[-top_10_percent_n:]

	top_25_percent_n = int(round(0.25 * len(sorted_by_fm)))
	top_25_percent   = sorted_by_fm[-top_25_percent_n:]

	bottom_10_percent = sorted_by_fm[:top_10_percent_n]
	bottom_25_percent = sorted_by_fm[:top_25_percent_n]

	show_histograms(top_2_5_percent,    "Top 2.5 Percent")
	show_histograms(top_5_percent,    "Top 5 Percent")
	show_histograms(top_10_percent,    "Top 10 Percent")
	# show_histograms(top_25_percent,    "Top 25 Percent")
	# show_histograms(bottom_25_percent, "Bottom 25 Percent")
	#show_histograms(bottom_10_percent, "Bottom 10 Percent")