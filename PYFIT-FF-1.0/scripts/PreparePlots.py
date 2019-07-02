import matplotlib.pyplot as plt
import matplotlib
import numpy             as np
import sys
import copy
import json
import os
import tl
from   mpldatacursor        import datacursor
from   scipy.optimize       import curve_fit
from   mpl_toolkits.mplot3d import Axes3D
from   matplotlib           import cm
from   matplotlib.colors    import ListedColormap, Normalize
import warnings
warnings.filterwarnings("ignore")
#plt.ion()

sigmax         = 0.05
sigmay         = 0.05
_tick_count_x  = None
_tick_count_y  = None
_title         = None
_levels        = None
_uniform       = False
_integer_ticks_x = False
_integer_ticks_y = False
_bin_min         = None
_bin_max         = None
_draw_trendline  = False

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

	grid = np.zeros((grid_size, grid_size))

	for yidx, yi in enumerate(y_rng):
		for xidx, xi in enumerate(x_rng):
			weights  = (1 / (2*np.pi*sigmax*sigmay))*np.exp(-(((x - xi)**2 / (2*sigmax**2)) + ((y - yi)**2 / (2*sigmay**2))))
			mean_val = np.average(z, weights=weights)
			grid[yidx][xidx] = mean_val

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

	ticksy = ticksx = ticks

	if _tick_count_x is not None:
		ticksx = _tick_count_x

	if _tick_count_y is not None:
		ticksy = _tick_count_y

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

	plot = ax.contour(
		values, 
		cmap=_cmap, 
		levels=levels, 
		linewidths=2.6
	)
	
	#_ticks = np.arange(0, grid_size + 1, grid_size // ticks)
	_ticks_x = np.linspace(0, grid_size - 1, ticksx)
	_ticks_x[_ticks_x > 0] -= 1
	


	_ticks_y = np.linspace(0, grid_size - 1, ticksy)
	_ticks_y[_ticks_y > 0] -= 1
	


	if _integer_ticks_x:
		# Round the tick locations to the nearest integer.
		_ticks_x = [int(round(t)) + 1 for t in _ticks_x]
		ax.set_xticks(_ticks_x)
		ax.set_xticklabels(['%i'%int(round(i)) for i in np.linspace(min(x), max(x), ticksx)], fontsize=20)
	else:
		ax.set_xticks(_ticks_x)
		ax.set_xticklabels(['%1.2f'%i for i in np.linspace(min(x), max(x), ticksx)], fontsize=20)


	if _integer_ticks_y:
		_ticks_y = [int(round(t)) + 1 for t in _ticks_y]
		ax.set_yticks(_ticks_y)
		ax.set_yticklabels(['%i'%int(round(i)) for i in np.linspace(min(y), max(y), ticksy)], fontsize=20)
	else:
		ax.set_yticks(_ticks_y)
		ax.set_yticklabels(['%1.2f'%i for i in np.linspace(min(y), max(y), ticksy)], fontsize=20)

	
	ax.set_xlabel(xlabel, fontsize=22)
	ax.set_ylabel(ylabel, fontsize=22)
	ax.xaxis.set_tick_params(width=2)
	ax.yaxis.set_tick_params(width=2)
	for axis in ['top','bottom','left','right']:
  		ax.spines[axis].set_linewidth(1.8)
	ax.set_title(title, fontsize=20)

	if show_points:
		# Normalize the values for the scatterplot points to between
		# zero and 1.0.
		cmap = matplotlib.cm.get_cmap(_cmap)

		xy = np.zeros((x.shape[0], 2))
		xy[:, 0] = x
		xy[:, 1] = y

		unique_points = np.unique(xy, axis=0)
		means         = np.zeros(unique_points.shape[0])

		for idx, [xi, yi] in enumerate(unique_points):
			# Find all points in the original arrays
			# that have the same location and average their
			# values.
			x_same       = x == xi
			y_same       = y == yi
			same_indices = x_same & y_same
			means[idx]   = z[same_indices].mean()


		# Normalize the values.
		means    = (means - means.min()) / (means.max() - means.min())
		x_scaled = unique_points[:, 0]
		y_scaled = unique_points[:, 1]

		x_scatter = (grid_size - 1) * ((x_scaled - x_scaled.min()) / (x_scaled.max() - x_scaled.min()))
		y_scatter = (grid_size - 1) * ((y_scaled - y_scaled.min()) / (y_scaled.max() - y_scaled.min()))

		# Move points in a little bit from the edge so they are visible.
		x_scatter[x_scatter <= 0] += 1
		x_scatter[x_scatter >= grid_size - 1] -= 1
		y_scatter[y_scatter <= 0] += 1
		y_scatter[y_scatter >= grid_size - 1] -= 1

		if _integer_ticks_y:
			y_scatter = [int(round(t)) for t in y_scatter]

		if _integer_ticks_x:
			x_scatter = [int(round(t)) for t in x_scatter]

		point_colors = cmap(means)
		ax.scatter(x_scatter, y_scatter, s=50, facecolor=point_colors, edgecolor='#FFFFFF')


	ax.set_xlim(0, grid_size - 1)
	ax.set_ylim(0, grid_size - 1)
	ax.set_aspect(aspect=1)
	lb = ax.clabel(plot, inline=1, fontsize=16, colors='#000000', manual=True, fmt='%1.3f')
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

def single_series_scatter(x, y, xlabel, ylabel, fit_lines=None):
	fig, ax = plt.subplots(1, 1)

	if fit_lines != None:
		ax.plot(fit_lines[0][0], fit_lines[0][1])

	ax.scatter(x, y, s=5)
	ax.set_title(_title or "Unspecified")
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	plt.show()


def multi_series_scatter(series, xlabel, ylabel, fit_lines=None):
	fig, ax = plt.subplots(1, 1)


	if _draw_trendline:

		for j, s in enumerate(series):
			tl_rng = np.unique(s[0])
			tl_y   = np.zeros(tl_rng.shape[0])
			# We want one point in the trendline for each x axis point in the data.

			for i, x in enumerate(tl_rng):
				y       = s[1][s[0] == x]
				tl_y[i] = y.mean()

			# Plot the trendline
			if j < len(preffered_plot_styles):
				plt.plot(tl_rng, tl_y, linewidth=1.0, **preffered_plot_styles[j])
			else:
				plt.plot(tl_rng, tl_y, linewidth=1.0)



		# For each x_point, 

	series_names = []
	series_objs  = []

	if fit_lines != None:
		for fit in fit_lines:
			pl, = ax.plot(fit[0], fit[1])
			series_names.append('fit for %s'%fit[2])
			series_objs.append(pl)

	for idx, s in enumerate(series):
		if idx < len(preffered_scatter_styles):
			pl = ax.scatter(s[0], s[1], **preffered_scatter_styles[idx])
		else:
			pl = ax.scatter(s[0], s[1], s=10)
		series_names.append(s[2])
		series_objs.append(pl)

	ax.legend(series_objs, series_names)
	ax.set_title(_title or "Unspecified")
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	plt.show()

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

def pcc(x, y):
	x = np.array(x)
	y = np.array(y)

	xm = x.mean()
	ym = y.mean()

	top    = ((x - xm)*(y - ym)).mean()
	bottom = x.std()*y.std()

	return top / bottom

def true_histogram_plot(to_show, bins, xlabel):
	fig, ax = plt.subplots(1, 1)
		
	ax.hist(
		to_show, 
		bins=bins, 
		edgecolor='black', 
		linewidth=0.5
	)

	ax.set_xlabel(xlabel)
	ax.set_ylabel("Quantity")

	ax.set_title(_title or "Unspecified")

	plt.show()

preffered_scatter_styles = [
	{'marker': 's', 'edgecolors': 'blue',    's': 40, 'facecolors': 'none'},
	{'marker': '^', 'edgecolors': 'magenta', 's': 40, 'facecolors': 'none'},
	{'marker': 'd', 'edgecolors': 'green',   's': 40, 'facecolors': 'none'},
	{'marker': 'h', 'edgecolors': 'orange',  's': 40, 'facecolors': 'none'},
	{'marker': '>', 'edgecolors': '#00bcd9', 's': 40, 'facecolors': 'none'},
	{'marker': '<', 'edgecolors': '#4a00c9', 's': 40, 'facecolors': 'none'}
]

preffered_plot_styles = [
	{'color' : 'blue',    'linestyle' : ':'},
	{'color' : 'magenta', 'linestyle' : '-.'},
	{'color' : 'green',   'linestyle' : '--'},
	{'color' : 'orange',     'linestyle' : '-'},
	{'color' : '#00bcd9',    'linestyle' : ':'},
	{'color' : '#4a00c9', 'linestyle' : '-.'}
]

if __name__ == '__main__':
	


	root_dir = sys.argv[1] # The directory that contains all of the idx_* directories

	args = [arg.lower() for arg in sys.argv[2:]]

	small_mode  = '--small-mode'  in args # Use if only .json files are available
	small_load  = '--small-load'  in args # Load only the small results file
	do_filter   = '--filter'      in args # Run the filter code below
	show_points = '--show-points' in args # Whether or not to show data points in 
	                                      # heatmaps
	load_error  = '--load-error'  in args # Load the error_log.txt and validation_loss_log.txt files.
	find_error_convergence = '--find-error-convergence' in args
	ignore_non_converged = '--ignore-non-convergence' in args
	_uniform    = '--uniform-histogram' in args

	fit_reciprocal = '--fit-reciprocal' in args
	fit_linear     = '--fit-linear'     in args

	_draw_trendline = '--draw-trendline' in args

	_integer_ticks_x = '--integer-ticks-x' in args
	_integer_ticks_y = '--integer-ticks-y' in args

	if ignore_non_converged:
		print("Using non-converged networks as well.")

	colormap = None
	if '--colormap' in args:
		colormap = sys.argv[args.index('--colormap') + 3]

	if colormap is not None:
		if os.path.isfile(colormap):
			# Load the custom colormap.
			f   = open(colormap, 'r')
			raw = f.read()
			f.close()

			lines = raw.split('\n')
			c = np.zeros((len(lines), 4))

			for idx, line in enumerate(lines):
				c[idx, :3] = [float(i) / 255.0 for i in line.split(' ')]
				c[idx,  3] = 1.0

			customRYG = ListedColormap(c)
			colormap  = customRYG

			

	if '--sigma' in args:
		sigmax = sigmay = float(sys.argv[args.index('--sigma') + 3])

	if '--title' in args:
		_title = sys.argv[args.index('--title') + 3]

	if '--levels' in args:
		_levels = int(sys.argv[args.index('--levels') + 3])

	if '--tick-count-x' in args:
		_tick_count_x = int(sys.argv[args.index('--tick-count-x') + 3])

	if '--tick-count-y' in args:
		_tick_count_y = int(sys.argv[args.index('--tick-count-y') + 3])

	if '--bin-range' in args:
		_bin_range = sys.argv[args.index('--bin-range') + 3]
		[_bin_min, _bin_max] = [loat(i) for i in _bin_range.split]

	export_template = None
	if '--re-export' in args:
		export_template = sys.argv[args.index('--re-export') + 3]

	plot_error = None
	if '--plot-error' in args:
		plot_error = sys.argv[args.index('--plot-error') + 3]


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

	true_histograms = []

	for arg in args:
		if arg.startswith('--true-histogram'):
			_, data, spec, ratio, sort, bins = arg.split(':')

			histogram = [
				data, 
				spec == 'bottom',
				float(ratio),
				sort,
				int(bins)
			]

			true_histograms.append(histogram)

	show_pcc = False
	show_fit = False
	scatter = []
	for arg in args:
		if arg.startswith('--scatter'):
			scatter_args = arg.split(':')[1:]

			if scatter_args[-1].startswith('pcc'):
				show_pcc     = True
				scatter_args = scatter_args[:-1]

			scatter.append([*scatter_args])


	criterion = None

	for arg in args:
		if arg.startswith('--filter'):
			_, expr = arg.split(':')
			expr = expr.replace('?', 'values')
			criterion = expr

	print_stmt = None
	for arg in args:
		if arg.startswith('--print'):
			_, expr = arg.split(':')
			expr = expr.replace('?', 'values')
			print_stmt = expr


	print("Analyzing %s"%root_dir)


	# ==================================================
	# Data Loading
	# ==================================================

	if load_error:

		print("Loading Error Files")
		e_loc, e_raw, e_stat = tl.walk_dir_recursive(
			root_dir, 6,
			name='loss_log.txt'
		)

		v_loc, v_raw, v_stat = tl.walk_dir_recursive (
			root_dir, 6,
			name='validation_loss_log.txt'
		)

		# Parse the files and pair them together
		# by their index.
		error_info = {}

		for t_loc, t_err in zip(e_loc, e_raw):
			# Figure out the index of this set.
			comp    = t_loc.split('/')
			idx_str = ''
			for c in comp:
				if c.startswith('idx_'):
					idx_str = c.split('_')[1]

			# Load the error information.
			e_values = [float(i) for i in t_err.split('\n') if i != '']

			# Find the corresponding validation info
			for _v_loc, v_err in zip(v_loc, v_raw):
				if idx_str in _v_loc:
					v_values = [float(i) for i in v_err.split('\n') if i != '']

			error_info[idx_str] = (e_values, v_values)

		first_elem = error_info[list(error_info.keys())[0]]

		validation_interval = len(first_elem[0]) // len(first_elem[1])

	

	if plot_error != None:
		data = error_info[plot_error]
		t_rng = np.arange(len(data[0]))
		v_rng = np.arange(len(data[1]))*validation_interval

		plt.scatter(t_rng, data[0], s=5)
		plt.scatter(v_rng, data[1], s=5)
		plt.show()

	results   = [] # master_results.json file data
	data      = [] # final_data.json file data
	locations = [] # directory the data is in

	divergent_results   = []
	divergent_locations = []

	for subdir in os.listdir(root_dir):
		new_dir = root_dir + subdir + '/'
		no_file = True
		if os.path.isdir(new_dir) and 'idx' in new_dir:
			if small_mode:
				contents_dir = new_dir
				results_file = contents_dir + 'master_results.json'
				if os.path.isfile(results_file):
					is_divergent, result, location, run_data = load_files(contents_dir, small_load)
					no_file = False
					
			else:
				for subsubdir in os.listdir(new_dir):
					contents_dir = new_dir + subsubdir + '/'
					results_file = contents_dir + 'master_results.json'
					if os.path.isfile(results_file):
						is_divergent, result, location, run_data = load_files(contents_dir, small_load)
						no_file = False

			if not no_file:
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

	if ignore_non_converged:

		results.extend(non_param_convergence)
		data.extend(non_param_convergence_data)
		locations.extend(non_param_convergence_locations)


		results.extend(non_net_convergence)
		data.extend(non_net_convergence_data)
		locations.extend(non_net_convergence_locations)



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

		if 'mean_val' in res['scores']:
			ovf   = res['scores']['mean_val'] / res['scores']['mean_rmse']
			t_err = res['scores']['mean_rmse']
			v_err = res['scores']['mean_val']
			n_r   = len(res['parameter_set']['r_0_values'])
			n_l   = len(res['parameter_set']['legendre_polynomials'])
			score = (-np.tanh(1.7*(ovf) - 4) + 1) / (t_err * v_err * n_l * (1 + 2*n_r))
			point.update({
				'mval'       : res['scores']['mean_val'],
				'minval'     : res['scores']['min_val'],
				'maxval'     : res['scores']['max_val'],
				'stdval'     : res['scores']['std_val'],
				'diff'       : res['scores']['mean_val'] - res['scores']['mean_rmse'],
				'overfit'    : ovf,
				'score'      : score
			})

		critical_data.append(point)


	# Evaluate the filter if there is one.
	if criterion != None:
		tmp = []
		for values in critical_data:
			if eval(criterion):
				tmp.append(values)
		print("Filter eliminated %i points"%(len(critical_data) - len(tmp)))
		critical_data = tmp
	

	if load_error:
		# Filter the error data as well.
		err_tmp = {}
		for k in error_info.keys():
			for point in critical_data:
				if k in point['loc']:
					err_tmp.update({k: error_info[k]})
					break
		error_info = err_tmp

	if print_stmt != None:
		for values in critical_data:
			eval("print(%s)"%print_stmt)

	if export_template is not None:
		os.mkdir("config_re_export/")
		# We need to take every configuration set that survived the filter 
		# insert its parameters into the template specified and write it
		# out into its own new config file.
		print("Exporting %i new config files . . ."%len(critical_data))

		with open(export_template, 'r') as f:
			raw = f.read()
			template = json.loads(raw)

		output_dir = '/home/ajr6/2019-07-01/deep-sweep-rerun/'

		with open('rerun.sh', 'w') as run_file:
			for idx, data in enumerate(critical_data):
				with open(data['loc'] + 'master_results.json', 'r') as f:
					json_data = json.loads(f.read())
					new_config = copy.deepcopy(template)
					new_config['parameter-set'] = json_data['parameter_set']

				with open("config_re_export/config_%05i.json"%(idx), 'w') as f:
					f.write(json.dumps(new_config))

				output = '%s%s'%(output_dir, 'idx_%05i'%(idx))
				run_file.write("./run_eval_enki_generic.sh 1 %s %s\n"%("config_re_export/config_%05i.json"%(idx), output))
				run_file.write("sleep 0.1\n")

		print("done")



	if find_error_convergence:
		if not load_error:
			print("Must specify --load-error if specifying --find-error-convergence")
			exit(1)

		convergence_points    = np.zeros(len(error_info.keys()))
		convergence_threshold = 0.00001
		convergence_window    = 200
		step                  = 20
		for idx, k in enumerate(error_info.keys()):
			error = np.array(error_info[k][0])

			end   = error.shape[0]
			start = end - convergence_window

			while start >= 0:
				if error[start:end].std() > convergence_threshold:
					convergence_points[idx] = start + step
					break

				end   -= step
				start -= step

		print("Convergence Info: ")
		print("\tmin:  %i"%convergence_points.min())
		print("\tmax:  %i"%convergence_points.max())
		print("\tmean: %f"%convergence_points.mean())
		print("\tstd:  %f"%convergence_points.std())


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
		'mrmse'   : "Root Mean Squared Training Error",
		'nf'      : "Number of Features",
		'minrmse' : "Minimum Root Mean Squared Training Error",
		'maxrmse' : "Maximum Root Mean Squared Training Error",
		'stdrmse' : "Root Mean Squared Training Error Standard Deviation",
		'mval'    : "Root Mean Squared Validation Error",
		'minval'  : "Minimum Root Mean Squared Validation Error",
		'maxval'  : "Maximum Root Mean Squared Validation Error",
		'stdval'  : "Root Mean Squared Validation Error Standard Deviation",
		'overfit' : "Validation Error to Training Error Ratio",
		'score'   : "Overall Score",
		'diff'    : 'Validation Error Minus Training Error'
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

	for spl in scatter:
		if len(spl) == 3:
			# Split the plot into multiple series,
			# each series being for values with a unique
			# value of spl[3]
			x_data = [(c[spl[0]], c[spl[2]]) for c in critical_data]
			y_data = [(c[spl[1]], c[spl[2]]) for c in critical_data]


			f_data = [c[spl[2]] for c in critical_data]

			unq = np.unique(f_data)

			x_lbl  = names[spl[0]]
			y_lbl  = names[spl[1]]

			series = []
			fits   = []
			for u in unq:
				x_filtered = np.array([x[0] for x in x_data if x[1] == u])
				y_filtered = np.array([y[0] for y in y_data if y[1] == u])
				if isinstance(u, float):
					name = '%s = %f'%(names[spl[2]], u)
				else:
					name = '%s = %s'%(names[spl[2]], u)

				if show_pcc:
					print("Pearson Coefficient (%10s): %1.3f"%(name, pcc(x_filtered, y_filtered)))
				
				series.append((x_filtered, y_filtered, name))

				
				if fit_reciprocal:
					model    = lambda x, a, b, c: (a / (x - b)) + c
					res, cov = curve_fit(model, x_filtered, y_filtered)
					fn       = lambda x: (res[0] / (x - res[1])) + res[2]

					print("f(∞) = %1.2f"%res[2])

					x_fit = np.linspace(x_filtered.min(), x_filtered.max(), 128)
					y_fit = fn(x_fit)
					fits.append([x_fit, y_fit, name])
				elif fit_linear:
					model    = lambda x, a, b: a*x + b
					res, cov = curve_fit(model, x_filtered, y_filtered)
					fn       = lambda x: res[0]*x + res[1]

					x_fit = np.linspace(x_filtered.min(), x_filtered.max(), 128)
					y_fit = fn(x_fit)
					fits.append([x_fit, y_fit, name])
				

			multi_series_scatter(series, x_lbl, y_lbl, fits)

		else:
			x_data = np.array([c[spl[0]] for c in critical_data])
			y_data = np.array([c[spl[1]] for c in critical_data])
			x_lbl  = names[spl[0]]
			y_lbl  = names[spl[1]]

			if show_pcc:
				print("Pearson Coefficient: %1.3f"%pcc(x_data, y_data))

			fits = None
			if fit_reciprocal:
				fits = []
				model    = lambda x, a, b, c: (a / (x - b)) + c
				res, cov = curve_fit(model, x_data, y_data)
				fn       = lambda x: (res[0] / (x - res[1])) + res[2]
				print("f(∞) = %1.2f"%res[2])

				x_fit = np.linspace(x_data.min(), x_data.max(), 128)
				y_fit = fn(x_fit)
				fits.append([x_fit, y_fit, None])
			elif fit_linear:
				fits = []
				model    = lambda x, a, b: a*x + b
				res, cov = curve_fit(model, x_data, y_data)
				fn       = lambda x: res[0]*x + res[1]

				x_fit = np.linspace(x_data.min(), x_data.max(), 128)
				y_fit = fn(x_fit)
				fits.append([x_fit, y_fit, None])

			single_series_scatter(x_data, y_data, x_lbl, y_lbl, fits)


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



	for [data, spec, ratio, sort, bins] in true_histograms:
		# Sort by the specified value.
		sort_vals = sorted(critical_data, key=lambda x: x[sort])
		n_select  = int(round(ratio * len(sort_vals)))

		use = None
		if not spec:
			use = sort_vals[-n_select:]
		else:
			use = sort_vals[:n_select]

		#loc, count = gen_bar([p[data] for p in use])
		to_show = [p[data] for p in use]
		xlabel  = names[data]

		true_histogram_plot(to_show, bins, xlabel)
