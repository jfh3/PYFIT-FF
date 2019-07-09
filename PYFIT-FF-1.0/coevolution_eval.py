#!/usr/bin/env python3

import matplotlib
#matplotlib.use('Agg')

from sys import path
path.append("subroutines")
path.append("scripts")

from TrainingSet         import TrainingSetFile, WriteTrainingSet
from NeuralNetwork       import NeuralNetwork
from ConfigurationParser import TrainingFileConfig
from FFCorrelationCalc   import FFCorrelationCalc
from FFHeatmap           import GenHeatmap
from FFScatterPlots      import GenFFScatterPlots
from EvalNN              import GetTrainingSetInstance, RunNetwork
from CFCorrelationCalc   import CFCorrelationCalc
from CFHeatmap           import GenCFHeatmap
from CFScatterPlots      import GenCFScatterPlots

import tl
import Util
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.pyplot
import datetime
import time
import os
import sys
import copy
import subprocess
import argparse

def run(cmdline, _async=False, wd=None):
	if wd != None:
		process = subprocess.Popen(cmdline.split(' '), stdout=subprocess.PIPE, cwd=wd)
	else:
		process = subprocess.Popen(cmdline.split(' '), stdout=subprocess.PIPE)

	if not _async:
		output, error = process.communicate()

		if error != '' and error != None:
			eprint("ERROR: %s"%error)
			sys.exit(1)
	else:
		return process

# This will copy pyfit-ff and all of its code into run_dir
# replace all values in Config.py with those in config and
# will run pyfit-ff with the specified parameters. If
# config_file_path is set, Config.py will be loaded from there.
def run_pyfit_with_config(config, params, run_dir, config_file_path=None, _async=False):
	if not os.path.isdir(run_dir):
		os.mkdir(run_dir)

	# First, copy pyfit-ff.py and subroutines/ to 
	# the run_dir.

	os.mkdir(run_dir + 'nn_backup')

	pyfit_path       = run_dir + "pyfit-ff.py"
	subroutines_path = run_dir
	config_path      = run_dir + "subroutines/Config.py"



	if config_file_path != None:
		config_to_load = config_file_path
	else:
		config_to_load = config_path

	if not os.path.isdir(subroutines_path):
		os.mkdir(subroutines_path)

	run("cp pyfit-ff.py %s"%(pyfit_path))
	run("cp -r subroutines/ %s"%(subroutines_path))

	# Now we load and parse the config file into a dictionary.
	# We then replace all values using the config dictionary 
	# supplied.

	f     = open(config_to_load, 'r')
	lines = f.read().split('\n')
	f.close()

	# Take all lines that aren't whitespace and don't start 
	# with '#'. Split them up at the equal sign and align
	# all of the equal signs for readability.

	cleaned    = []
	for line in lines:
		if not line.isspace() and line != '' and line.lstrip()[0] != '#':
			cells = line.lstrip().rstrip().split('=')
			cleaned.append(cells)

	config_file_dict = {}
	for param, value in cleaned:
		config_file_dict[param] = value

	for key in config.keys():
		config_file_dict[key] = config[key]

	# Write this to Config.py
	f = open(config_path, 'w')
	f.write('\n'.join([k + ' = ' + v for k, v in config_file_dict.items()]))
	f.close()

	current_dir = os.getcwd()

	# Now that pyfit is copied and the config file is ready,
	# run pyfit with the specified parameters.
	if not _async:
		os.chdir(run_dir)
		run("python3 pyfit-ff.py %s"%(params))
		os.chdir(current_dir)
	else:
		return run("python3 pyfit-ff.py %s"%(params), _async=True, wd=run_dir)

def wait_for_processes(process_list, poll_interval=0.05):
	while True:
		all_done = True
		for process in process_list:
			if process.poll() == None:
				all_done = False

		if all_done:
			return
		else:
			time.sleep(poll_interval)

def pcc(a, b):
	a = np.array(a)
	b = np.array(b)

	a_mean = a.mean()
	b_mean = b.mean()

	top    = ((a - a_mean)*(b - b_mean)).mean()
	bottom = a.std() * b.std()

	return top / bottom 

# This will load the group error into a dictionary where each key
# is a group name and each value is a numpy array of the error for
# that group in order from first to last iteration.
def load_group_error(fname):
	with open(fname, 'r') as file:
		raw = file.read()

	lines  = raw.split('\n')
	header = lines[0]
	lines  = lines[1:]

	group_names = header.split(' ')[1:]
	n_groups    = len(group_names)

	rows = []
	for line in lines:
		if line != '':
			rows.append([float(i) for i in line.split(' ')[1:]])

	result = {}

	for g in range(n_groups):
		result[group_names[g]] = np.zeros(len(rows))

	for i, row in enumerate(rows):
		for j, cell in enumerate(row):
			result[group_names[j]][i] = cell

	return result

def graph_correlations(matrix, fname, title, delta_mode=False):
	# We should now have a proper correlation matrix. 
	# Construct an image for it.

	color_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 3))

	minval = matrix.min()
	maxval = matrix.max()
	rng    = maxval - minval

	for m in range(matrix.shape[0]):
		for n in range(matrix.shape[1]):
			val     = matrix[m][n]
			r, g, b = 1.0, 1.0, 1.0

			if delta_mode:
				val = (val - minval) / rng
			
			if val > 0.0:
				r -= val
				b -= val
			else:
				g += val
				b += val

			if m == n:
				r, g, b = 0.0, 0.0, 0.0
			color_matrix[m][n][0] = r
			color_matrix[m][n][1] = g
			color_matrix[m][n][2] = b

	# Now that we have a color matric, graph it.
	fig, ax = plt.subplots()
	pl = ax.imshow(color_matrix, interpolation='none')

	ax.set_xticks(list(range(matrix.shape[1])))
	ax.set_yticks(list(range(matrix.shape[0])))
	ax.set_xticklabels(all_groups)
	ax.set_yticklabels(all_groups)

	for m in range(matrix.shape[0]):
		for n in range(matrix.shape[1]):
			ax.text(n, m, '%i'%(int(round(100*matrix[m][n]))), ha="center", va="center", color="black", fontsize=2)

	
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(3.0)

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(3.0)

	fig.subplots_adjust(bottom=0.2)

	plt.title(title)

	matplotlib.use('Agg')
	plt.savefig(fname, dpi=300)

	#matplotlib.use("TkAgg")
	#plt.show()

# Note: The majority of the configuration that defines how things
#        will run is defined in Config.py.
if __name__ == '__main__':
	# This program needs the following arguments:
	#     1) The folder to create to put output files into.
	#     2) The training set file that is being used to train the networks.
	#     3) The error target to give to each subgroup when it is the target.
	#     4) The default error to give to each subgroup when it is not the target.
	#     5) The file to use as Config.py for the pyfit runs.

	parser = argparse.ArgumentParser(
		description="For the given training set, trains one network per structural subgroup. During each training, " +
		"sets one target group to a lower error than the rest and determines the correlation between the other groups and " +
		"this group."
	)
	parser.add_argument(
		'-o', '--output-file', dest='opath', type=str, required=True, 
		help='The directory to output files to. Must not exist when program is started.'
	)
	parser.add_argument(
		'-t', '--training-set', dest='tfile', type=str, required=True, 
		help='The training set file to read group names from and that is being used for training. Should also be set in config file.'
	)
	parser.add_argument(
		'-e', '--error-target', dest='target_error', type=float, required=True, 
		help='The target error for the group being trained to low error.'
	)
	parser.add_argument(
		'-d', '--error-default', dest='default_error', type=float, required=True, 
		help='The default error target for all groups except for the group being studied.'
	)
	parser.add_argument(
		'-c', '--config-file', dest='config_file', type=str, required=True, 
		help='The file to use as Config.py for all pyfit runs. Some values are overridden by this script.'
	)
	parser.add_argument(
		'-p', '--pearson-correlation', dest='use_pearson', action='store_true',
		help='Whether or not to use the Pearson correlation metric.'
	)
	parser.add_argument(
		'-l', '--delta-correlation', dest='use_delta', action='store_true',
		help='Whether or not to use the delta correlation metric.'
	)
	parser.add_argument(
		'-s', '--skip-first', dest='skip_first', type=int, required=False, default=0,
		help='The number of training iterations from the beginning to skip for all calculations.'
	)
	parser.add_argument(
		'-i', '--training-iterations', dest='training_iterations', type=int, required=False, default=0,
		help='The number of training iterations to use. Default is whatever is in the config file.'
	)
	parser.add_argument(
		'-g', '--groups', dest='groups', type=str, required=False, nargs='*',
		help='The specific groups to target for this run. The default is to use all groups.'
	)

	args = parser.parse_args(sys.argv[1:])

	if args.groups == []:
		args.groups = None

	# Number of training iterations to skip from the beginning.
	#skip_first = 0

	#opath = sys.argv[1]

	if args.opath[-1] != '/':
		args.opath += '/'

	if os.path.isdir(args.opath):
		print("The output directory already exists.")
		exit(1)

	os.mkdir(args.opath)

	if not os.path.isfile(args.tfile):
		print("The specified training input file does not exist.")
		exit(1)



	if not os.path.isfile(args.config_file):
		print("The specified config file does not exist.")
		exit(1)



	if not args.use_pearson and not args.use_delta:
		print("Neither pearson, nor delta was specified.")

	# We need to load the training set file and get a list of structural
	# subgroups from it. Once this is complete, we run pyfit once for each
	# subgroup. Each time, we set the target error of that subgroup to be
	# lower than the rest. We then store the resulting per-group error
	# elsewhere for processing at the end.

	Util.init('log.txt')

	# Load whichever training set is specified above.
	training_set = TrainingSetFile(args.tfile)

	if args.groups is None:
		all_groups = []
		for struct_id in training_set.training_structures:
			all_groups.append(training_set.training_structures[struct_id][0].group_name)
	else:
		all_groups = args.groups


	# Filter out unique names.
	tmp = []
	for g in all_groups:
		if g not in tmp:
			tmp.append(g)

	# We want everything to be sorted so that all results have consistent ordering.
	all_groups = sorted(tmp)

	print("Performing analysis for %i structural groups . . . "%len(all_groups))

	# Iterate over the groups, each time reconfiguring which group has the lowest
	# target error.

	result_file_paths = []
	for group in all_groups:
		# Generate a directory to do the pyfit run in.
		run_dir  = args.opath   + 'pyfit_run_%s/'%group
		out_file = run_dir + 'gerr.txt' 
		os.mkdir(run_dir)

		# Generate a config dictionary to use for the run.
		
		config_for_run = {
			'DEFAULT_TARGET'   : '%f'%args.default_error,
			'GROUP_ERROR_FILE' : '\'gerr.txt\'',
			'SUBGROUP_TARGETS' : '{\'%s\':%f}'%(group, args.target_error)
		}

		if args.training_iterations != 0:
			config_for_run['MAXIMUM_TRAINING_ITERATIONS'] = '%i'%args.training_iterations

		print("Running Pyfit")
		run_pyfit_with_config(
			config_for_run,
			'-t -u',
			run_dir,
			config_file_path=args.config_file
		)

		result_file_paths.append(out_file)

	group_error_structs = {}
	for i, group in enumerate(all_groups):
		# Load the group error file for it.
		error = load_group_error(result_file_paths[i])
		group_error_structs[group] = error

	if args.use_pearson:
		print("Calculating Pearson Correlations")
		# We should now have a group error file for each subgroup.
		# We need to load all of them and calculate pearson correlation for
		# all of them.

		pearson_correlations = np.ones((len(all_groups), len(all_groups)))

		

		for i, group in enumerate(all_groups):
			# Load the group error file for it.
			error = group_error_structs[group]

			# Iterate over every group and calculate the correlation
			# between this group, and the other group.

			for j, other_group in enumerate(all_groups):
				pearson_correlations[i][j] = pcc(error[group][args.skip_first:], error[other_group][args.skip_first:])

		
		graph_correlations(pearson_correlations, args.opath + 'pcc.png', "Pearson Correlation Matrix")

		tl.ndarray_to_file(pearson_correlations, args.opath + 'pearson.mat')


	if args.use_delta:
		print("Calculating Delta Correlations")

		delta_correlations = np.ones((len(all_groups), len(all_groups)))

		# Unlike the pearson correlations, here we just consider
		# the ratio of the everage error between the two.
		for i, group in enumerate(all_groups):
			# Load the group error file for it.
			error = group_error_structs[group]

			# Iterate over every group and calculate the correlation
			# between this group, and the other group.

			for j, other_group in enumerate(all_groups):
				other_delta = np.abs(error[other_group][args.skip_first:].mean() - args.default_error)
				this_delta  = np.abs(error[group][args.skip_first:].mean()       - args.default_error)
				delta_correlations[i][j] = other_delta / this_delta

		graph_correlations(delta_correlations, args.opath + 'delta.png', "Delta Correlation Matrix")
		tl.ndarray_to_file(delta_correlations, args.opath + 'delta.mat')

	with open(args.opath + 'data.json', 'w') as file:
		tmp = {}
		for k in group_error_structs:
			tmp[k] = {}
			for k2 in group_error_structs[k]:
				tmp[k][k2] = group_error_structs[k][k2].tolist()
		file.write(json.dumps(tmp))
