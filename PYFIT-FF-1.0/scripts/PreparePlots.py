# Good ColorMaps:
#    seismic
#    inferno
#    
# --small-mode --small-load 
# Good Command Lines to try:
#    --heatmap:nf:ff:fm --colormap inferno
#    --heatmap:lmin:lmax:fm --colormap Accent
#    --heatmap:rmin:rmax:fm --colormap Accent
#    --heatmap:ff:fc:fm --colormap Accent
#    --heatmap:ff:fc:mrmse --colormap Accent
#    --heatmap:s:nf:fm --colormap gnuplot
#    --heatmap:nf:fc:fm --colormap gnuplot
#    --heatmap:nf:fc:fm --colormap Accent --show-points
#    --contour:ff:fc:fm --colormap cool
#    --contour:lmax:lmin:fm --colormap cool
#    --contour:rmax:rmin:fm --colormap cool
#    --contour:ff:fc:mrmse --colormap cool --show-points
#    --contour:nf:fc:fm --colormap cool --sigma 0.13
#    --contour:r#:l#:fm --colormap cool --sigma 0.08 --show-points
#    --contour:rmax:lmax:fm --colormap cool --sigma 0.08
#    --contour:rmin:lmin:fm --colormap cool --sigma 0.08 --show-points
#    --contour:ff:fc:fm --colormap cool --sigma 0.08
#    --contour:nf:fc:fm --colormap cool --sigma 0.08

import matplotlib.pyplot as plt
import matplotlib
import numpy             as np
import sys
import json
import os
from   mpldatacursor        import datacursor
from   mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
#plt.ion()

sigmax         = 0.05
sigmay         = 0.05
_tick_count    = None
_title         = None
_levels        = None
_uniform       = False
_integer_ticks = False

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
# It constructs a 3 dimensional array suitable for a matplotlib imshow plot.
# It does this by generating a grid of uniform points covering the full 
# domain of both the x an y axis. For each point in this grid, it constructs
# an average of the z value for all points within "cutoff" of this point. 
# It weights this average using a gaussian function of the distance that
# each of those points is from the current point. "sigmax" and "sigmay"
# correspond to the sigma parameters in this 2-D gaussion. 
# In essence, this function makes an attempt at Gaussian averaging of points
# in order to create a smooth uniform plot.
def xyz_to_img(x, y, z, cutoff=150.0, grid_size=150):
	global sigmax
	global sigmay

	x_rng = np.linspace(min(x), max(x), grid_size)
	y_rng = np.linspace(min(y), max(y), grid_size)

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
				print('not enough data')
			else:
				mean_val = np.average(z_cutoff, weights=weights)
				row.append(mean_val)
			#row.append(yi)
		grid.append(row)

	return grid


def triple_heatmap(x, y, z, xlabel, ylabel, title, grid_size=150, ticks=10, show_points=False, colormap=None):
	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(8, 8)

	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	values = xyz_to_img(x, y, z, grid_size=grid_size)

	x_rng = max(x) - min(x)
	y_rng = max(y) - min(y)

	if colormap is None:
		_cmap = 'inferno'
	else:
		_cmap = colormap

	plot = ax.imshow(values, cmap=_cmap, interpolation='bicubic')
	_ticks = np.arange(0, grid_size + 1, grid_size // ticks)
	_ticks[_ticks > 0] -= 1
	ax.set_xticks(_ticks)
	ax.set_yticks(_ticks)
	ax.set_xticklabels(['%1.2f'%i for i in np.linspace(min(x), max(x), ticks + 1)])
	ax.set_yticklabels(['%1.2f'%i for i in np.linspace(min(y), max(y), ticks + 1)])
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)

	if show_points:
		# We need to remap these points to match the current scaling.
		x = (grid_size - 1) * ((x - min(x)) / x_rng)
		y = (grid_size - 1) * ((y - min(y)) / y_rng)
		x[x <= min(x)] += 1
		x[x >= max(x)] -= 1
		y[y <= min(y)] += 1
		y[y >= max(y)] -= 1
		ax.scatter(x, y, s=5, facecolor='#000000', edgecolor='#FFFFFF')


	
	ax.set_xlim(0, grid_size - 1)
	ax.set_ylim(0, grid_size - 1)
	fig.colorbar(plot)
	ax.set_aspect(aspect=1)
	plt.show()

def triple_contour(x, y, z, xlabel, ylabel, title, grid_size=150, ticks=10, levels=10, show_points=False, colormap=None):
	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(8, 8)

	if _tick_count is not None:
		ticks = _tick_count

	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	values = xyz_to_img(x, y, z, grid_size=grid_size)

	x_rng = max(x) - min(x)
	y_rng = max(y) - min(y)

	if colormap is None:
		_cmap = 'inferno'
	else:
		_cmap = colormap

	plot = ax.contour(values, cmap=_cmap, levels=levels, linewidths=2.6)
	
	if _integer_ticks:
		_ticks = np.arange(0, grid_size + 1, grid_size // ticks)
		_ticks[_ticks > 0] -= 1
		ax.set_xticks(_ticks)
		ax.set_yticks(_ticks)
		ax.set_xticklabels(['%i'%int(round(i)) for i in np.linspace(min(x), max(x), ticks + 1)], fontsize=16)
		ax.set_yticklabels(['%i'%int(round(i)) for i in np.linspace(min(y), max(y), ticks + 1)], fontsize=16)
		
	else:
		_ticks = np.arange(0, grid_size + 1, grid_size // ticks)
		_ticks[_ticks > 0] -= 1
		ax.set_xticks(_ticks)
		ax.set_yticks(_ticks)
		ax.set_xticklabels(['%1.2f'%i for i in np.linspace(min(x), max(x), ticks + 1)], fontsize=16)
		ax.set_yticklabels(['%1.2f'%i for i in np.linspace(min(y), max(y), ticks + 1)], fontsize=16)

	ax.set_xlabel(xlabel, fontsize=20)
	ax.set_ylabel(ylabel, fontsize=20)
	ax.xaxis.set_tick_params(width=2)
	ax.yaxis.set_tick_params(width=2)
	for axis in ['top','bottom','left','right']:
  		ax.spines[axis].set_linewidth(1.8)
	ax.set_title(title, fontsize=20)

	if show_points:
		# Normalize the values for the scatterplot points to between
		# zero and 1.0.
		values       = (z - z.min()) / (z.max() - z.min())
		cmap         = matplotlib.cm.get_cmap(colormap)
		# point_colors = [cmap(v) for v in values]


		# We need to remap these points to match the current scaling.
		x = (grid_size - 1) * ((x - min(x)) / x_rng)
		y = (grid_size - 1) * ((y - min(y)) / y_rng)
		x[x <= min(x)] += 1
		x[x >= max(x)] -= 1
		y[y <= min(y)] += 1
		y[y >= max(y)] -= 1

		# Now we get all of the points that are basically on top of eachother and
		# average them.
		new_colors = []
		for xi, yi in zip(x, y):
			# Get all of the points that are at the same location.
			x_same = (np.abs(x - xi) < (x_rng / 100))
			y_same = (np.abs(y - yi) < (y_rng / 100))

			match_indices = x_same & y_same
			avg           = values[match_indices].mean()
			new_colors.append(cmap(avg**(1.85)))

		point_colors = new_colors


		ax.scatter(x, y, s=50, facecolor=point_colors, edgecolor='#FFFFFF')


	ax.set_xlim(0, grid_size - 1)
	ax.set_ylim(0, grid_size - 1)
	ax.set_aspect(aspect=1)
	lb = ax.clabel(plot, inline=1, fontsize=15, colors='#000000', manual=True, fmt='%1.2f')
	for l in lb:
		l.set_fontweight('bold')
	plt.show()

def generic_3d(_x, _y, _z, points, show_points=False, colormap=None, names=None, contour=False):
	x = np.array([p[_x] for p in points])
	y = np.array([p[_y] for p in points])
	z = np.array([p[_z] for p in points])


	if names is not None:
		xlabel = names[_x]
		ylabel = names[_y]
		if _title is None:
			title  = '%s\n as a Function of %s\n and %s'%(names[_z], xlabel, ylabel)
		else:
			title = _title
	else:
		xlabel = 'Unspecified'
		ylabel = 'Unspecified'
		title  = 'Unspecified'

	if contour:
		levels = 10
		if _levels is not None:
			levels = _levels

		triple_contour(
			x, y, z, 
			xlabel, 
			ylabel, 
			title,
			show_points=show_points,
			colormap=colormap,
			levels=levels
		)
	else:
		triple_heatmap(
			x, y, z, 
			xlabel, 
			ylabel, 
			title,
			show_points=show_points,
			colormap=colormap
		)

def histogram_plot(locations, counts, xlabel):
	fig, ax = plt.subplots(1, 1)

	width = (max(locations) - min(locations)) / (len(locations)*1.8)

	if _uniform:
		_loc = np.linspace(min(locations), max(locations), len(locations))

		ax.bar(_loc, counts, edgecolor='black', linewidth=0.5, width=width)
	else:
		ax.bar(locations, counts, edgecolor='black', linewidth=0.5, width=width)

	ax.set_xlabel(xlabel)
	ax.set_ylabel("Quantity")

	if _uniform:
		ax.set_xticks(_loc)
	else:
		ax.set_xticks(locations)

	ax.set_xticklabels(['%1.2f'%l for l in locations], rotation = 45)
	ax.set_title(_title or "Unspecified")

	plt.show()

if __name__ == '__main__':
	root_dir = sys.argv[1] # The directory that contains all of the idx_* directories

	args = [arg.lower() for arg in sys.argv[2:]]

	small_mode  = '--small-mode'  in args # Use if only .json files are available
	small_load  = '--small-load'  in args # Load only the small results file
	do_filter   = '--filter'      in args # Run the filter code below
	show_points = '--show-points' in args # Whether or not to show data points in 
	                                      # heatmaps
	_uniform    = '--uniform-histogram' in args

	_integer_ticks = '--integer-ticks' in args

	colormap = None
	if '--colormap' in args:
		colormap = sys.argv[args.index('--colormap') + 3]

	if '--sigma' in args:
		sigmax = sigmay = float(sys.argv[args.index('--sigma') + 3])

	if '--title' in args:
		_title = sys.argv[args.index('--title') + 3]

	if '--levels' in args:
		_levels = int(sys.argv[args.index('--levels') + 3])

	if '--tick-count' in args:
		_tick_count = int(sys.argv[args.index('--tick-count') + 3])

	# Look for a heatmap argument in the form:
	#    --heatmap:x:y:z, where x, y and z are 
	#    keys in the critical data array.

	heatmaps = []

	for arg in args:
		if arg.startswith('--heatmap'):
			_, x, y, z = arg.split(':')
			heatmaps.append((x, y, z, False))
		elif arg.startswith('--contour'):
			_, x, y, z = arg.split(':')
			heatmaps.append((x, y, z, True))

	histograms = []

	for arg in args:
		if arg.startswith('--histogram'):
			_, data, spec, ratio, sort = arg.split(':')

			histogram = [
				data, 
				spec == 'bottom',
				float(ratio),
				sort
			]

			histograms.append(histogram)


	criterion = None

	for arg in args:
		if arg.startswith('--filter'):
			_, expr = arg.split(':')
			expr = expr.replace('%', 'values')
			criterion = expr


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
		r_density = len(res['parameter_set']['r_0_values'])
		r_rng     = np.array(res['parameter_set']['r_0_values']).max()
		r_rng    -= np.array(res['parameter_set']['r_0_values']).min()
		r_density /= r_rng


		l_density = len(res['parameter_set']['legendre_polynomials'])
		l_rng     = np.array(res['parameter_set']['legendre_polynomials']).max()
		l_rng    -= np.array(res['parameter_set']['legendre_polynomials']).min()
		l_density /= l_rng

		point = {
			'loc'     : loc,
			'r'       : res['parameter_set']['r_0_values'],
			'r#'      : len(res['parameter_set']['r_0_values']),
			'rd'      : r_density,
			'rm'      : np.array(res['parameter_set']['r_0_values']).mean(),
			'rmin'    : np.array(res['parameter_set']['r_0_values']).min(),
			'rmax'    : np.array(res['parameter_set']['r_0_values']).max(),
			'l'       : res['parameter_set']['legendre_polynomials'],
			'l#'      : len(res['parameter_set']['legendre_polynomials']),
			'ld'      : l_density,
			'lm'      : np.array(res['parameter_set']['legendre_polynomials']).mean(),
			'lmin'    : np.array(res['parameter_set']['legendre_polynomials']).min(),
			'lmax'    : np.array(res['parameter_set']['legendre_polynomials']).max(),
			's'       : res['parameter_set']['gi_sigma'],
			'fm'      : res['scores']['figure_of_merit'],
			'ff'      : res['scores']['mean_ff_correlation'],
			'fc'      : res['scores']['mean_fc_correlation'],
			'mrmse'   : res['scores']['mean_rmse'],
			'nf'      : len(res['parameter_set']['r_0_values']) * len(res['parameter_set']['legendre_polynomials']),
			'minrmse' : res['scores']['min_rmse'],
			'maxrmse' : res['scores']['max_rmse'],
			'stdrmse' : res['scores']['std_rmse']
		}

		critical_data.append(point)


	# Evaluate the filter if there is one.
	if criterion != None:
		tmp = []
		for values in critical_data:
			if eval(criterion):
				tmp.append(values)
		print("Filter eliminated %i points"%(len(critical_data) - len(tmp)))
		critical_data = tmp
	


	names = {
		'r'       : "$r_0$ Values",
		'r#'      : "Number of $r_0$ Values",
		'rd'      : "Density of $r_0$ Values",
		'rm'      : "Mean $r_0$ Value",
		'rmin'    : "Minimum $r_0$ Value",
		'rmax'    : "Maximum $r_0$ Value",
		'l'       : "Legendre Polynomials",
		'ld'      : "Density of Legendre Polynomial Orders",
		'l#'      : "Number of Legendre Polynomials",
		'lm'      : "Mean Legendre Polynomial Order",
		'lmin'    : "Minimum Legendre Polynomial Order",
		'lmax'    : "Maximum Legendre Polynomial Order",
		's'       : "Sigma",
		'fm'      : "Figure of Merit",
		'ff'      : "Mean Feature - Feature Correlation",
		'fc'      : "Mean Feature - Output Correlation",
		'mrmse'   : "Mean Root Mean Squared Training Error",
		'nf'      : "Number of Features",
		'minrmse' : "Minimum Root Mean Squared Training Error",
		'maxrmse' : "Maximum Root Mean Squared Training Error",
		'stdrmse' : "Root Mean Squared Training Error Standard Deviation"
	}

	if colormap is not None:
		print("Using colormap: %s"%colormap)

	if len(heatmaps) != 0:
		print("Constructing the following 3d plots:")
		for h in heatmaps:
			print("\tx :: %s"%(names[h[0]]))
			print("\ty :: %s"%(names[h[1]]))
			print("\tz :: %s"%(names[h[2]]))
			print('')

	for h in heatmaps:
		generic_3d(
			h[0], h[1], h[2], 
			critical_data,
			show_points=show_points, 
			colormap=colormap, 
			names=names,
			contour=h[3]
		)

	def gen_bar(vals):
		locations = np.unique(vals).tolist()
		vals      = np.array(vals)
		counts    = []

		for location in locations:
			counts.append(len(vals[(vals == location)]))

		return locations, counts

	for [data, spec, ratio, sort] in histograms:
		# Sort by the specified value.
		sort_vals = sorted(critical_data, key=lambda x: x[sort])
		n_select  = int(round(ratio * len(sort_vals)))

		use = None
		if not spec:
			use = sort_vals[-n_select:]
		else:
			use = sort_vals[:n_select]

		loc, count = gen_bar([p[data] for p in use])
		xlabel = names[data]

		histogram_plot(loc, count, xlabel)

