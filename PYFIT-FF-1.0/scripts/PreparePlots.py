import matplotlib.pyplot as plt
import numpy             as np
import sys
import json
import os
from   mpldatacursor        import datacursor
from   mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

def load_files(path, minimal=False):
	results_file = path + 'master_results.json'
	data_file    = path + 'final_data.json'

	f = open(results_file, 'r')
	raw = f.read()
	f.close()

	if not minimal:
		f = open(data_file, 'r')
		raw2 = f.read()
		f.close()

	if 'NaN' in raw or 'Infinity' in raw:
		to_parse = raw.replace('NaN', '0.0').replace('Infinity', '0.0')
		div  = json.loads(to_parse)
		file = results_file
		del to_parse

		return True, div, file, None
	else:
		res  = json.loads(raw)
		file = contents_dir
		if not minimal:
			data = json.loads(raw2)
		else:
			data = None

		# Too much memory is being taken up. Need to delete some keys.


		return False, res, file, data

def sort_by(l, k, r=False):
	return sorted(l, key = lambda x: x[k], reverse = r)

# This function takes a set of x, y and z points as its primary arguments.
# 
def xyz_to_img(x, y, z, sigmax=0.1, sigmay=0.1, cutoff=1.0):
	x_rng = np.linspace(min(x), max(x), 100)
	y_rng = np.linspace(min(y), max(y), 100)

	sigmax = (x_rng[-1] - x_rng[0])*sigmax
	sigmay = (y_rng[-1] - y_rng[0])*sigmay

	x = np.array(x)
	y = np.array(y)
	z = np.array(z)

	grid = []

	for yi in y_rng:
		row = []
		for xi in x_rng:
			distances = np.sqrt((xi - x)**2 + (y - yi)**2)
			
			x_cutoff = x[distances <= cutoff] - xi
			y_cutoff = y[distances <= cutoff] - yi
			z_cutoff = z[distances <= cutoff]
			weights       = (1 / (2*np.pi*sigmax*sigmay))*np.exp(-(((x_cutoff)**2 / (2*sigmax**2)) + ((y_cutoff)**2 / (2*sigmay**2))))
			

			if weights.sum() == 0.0:
				row.append(0.0)
			else:
				mean_val = np.average(z_cutoff, weights=weights)
				row.append(mean_val)
			#row.append(yi)
		grid.append(row)

	return grid


def r_l_fm_heatmap(points):
	fig, ax = plt.subplots(1, 1)

	x = [p['rm'] for p in points]
	y = [p['lm'] for p in points]
	z = [p['fm'] for p in points]
	values = xyz_to_img(x, y, z)

	plot = ax.imshow(values, cmap='Blues', interpolation='bicubic')
	ax.set_xticks(np.arange(0, 100, 10))
	ax.set_yticks(np.arange(0, 100, 10))
	ax.set_xticklabels(['%1.1f'%i for i in np.linspace(min(x), max(x), 10)])
	ax.set_yticklabels(['%1.1f'%i for i in np.linspace(min(y), max(y), 10)])
	ax.set_xlabel('Mean $r_0$ Value')
	ax.set_ylabel('Mean Legendre Order')
	ax.set_title('Figure of Merit as a Function of Mean Legendre order and Mean $r_0$ Value')
	fig.colorbar(plot)
	ax.set_aspect(aspect=1)
	plt.show()

if __name__ == '__main__':
	root_dir = sys.argv[1] # The directory that contains all of the idx_* directories

	args = [arg.lower() for arg in sys.argv[2:]]

	small_mode = '--small-mode' in args # Use if only .json files are available
	small_load = '--small-load' in args # Load only the small results file
	do_filter  = '--filter'     in args # Run the filter code below

	print("Analyzing %s"%root_dir)


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
		if os.path.isdir(new_dir) and 'idx' in new_dir:
			if small_mode:
				contents_dir = new_dir
				results_file = contents_dir + 'master_results.json'
				if os.path.isfile(results_file):
					is_divergent, result, location, run_data = load_files(contents_dir, small_load)
					
			else:
				for subsubdir in os.listdir(new_dir):
					contents_dir = new_dir + subsubdir + '/'
					results_file = contents_dir + 'master_results.json'
					if os.path.isfile(results_file):
						is_divergent, result, location, run_data = load_files(contents_dir, small_load)

			if is_divergent:
				divergent_results.append(result)
				divergent_locations.append(location)
			else:
				results.append(result)
				locations.append(location)
				data.append(run_data)

	# composite_sort = [(a, b, c) for a, b, c in zip(results, data, locations)]
	# composite_sort = sorted(composite_sort, key=lambda x: x[2])

	# results   = [i[0] for i in composite_sort]
	# data      = [i[1] for i in composite_sort]
	# locations = [i[2] for i in composite_sort]

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

	# Now we prepare a minimal list of parameters that can be processed
	# easily.

	critical_data = []
	for res, loc in zip(results, locations):
		point = {
			'loc'   : loc,
			'r'     : res['parameter_set']['r_0_values'],
			'r#'    : len(res['parameter_set']['r_0_values']),
			'rm'    : np.array(res['parameter_set']['r_0_values']).mean(),
			'l'     : res['parameter_set']['legendre_polynomials'],
			'l#'    : len(res['parameter_set']['legendre_polynomials']),
			'lm'    : np.array(res['parameter_set']['legendre_polynomials']).mean(),
			's'     : res['parameter_set']['gi_sigma'],
			'fm'    : res['scores']['figure_of_merit'],
			'ff'    : res['scores']['mean_ff_correlation'],
			'fc'    : res['scores']['mean_fc_correlation'],
			'mrmse' : res['scores']['mean_rmse']
		}

		critical_data.append(point)

	r_l_fm_heatmap(critical_data)