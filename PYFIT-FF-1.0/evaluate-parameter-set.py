# This program takes a file as the first argument, the configuration 
# parameters json file. It's second argument is the folder to create and store 
# the output information in. It is expected to not exist at the time that the 
# program is called. It will generate neural network files, LSParam files, 
# feature-feature correlations, heatmaps, scatterplot pngs, trained neural 
# networks, feature-output correlations, more scatterplot pngs and more 
# heatmaps, as well as actually calculating a final figure of merit score.

from sys import path
path.append("subroutines")

from TrainingSet         import TrainingSetFile, WriteTrainingSet
from NeuralNetwork       import NeuralNetwork
from ConfigurationParser import TrainingFileConfig

import json
import numpy as np
import matplotlib.pyplot
import datetime
import time
import os
import sys
import copy
import subprocess

# Print to stderr.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Reads and ensures the validity of the arguments.
# Returns a configuration dictionary and the output
# directory, ensuring that the output directory is
# empty and exists.
#
# return configuration, output_directory
def parse_and_validate_args():
	if len(sys.argv) != 3:
		eprint("This program takes 3 arguments.")
		sys.exit(1)

	configuration_file = sys.argv[1]

	if not os.path.isfile(configuration_file):
		eprint("The first argument should be a json file.")
		eprint("The file specified does not exist or isn't a file.")
		sys.exit(1)

	try:
		file = open(configuration_file)
		raw = file.read()
		file.close()
	except Exception as ex:
		eprint(str(ex))
		eprint("The configuration file exists but was unreadable.")
		sys.exit(1)

	try:
		configuration = json.loads(raw)
	except Exception as ex:
		eprint(str(ex))
		eprint("Could not parse the contents of the configuration file.")
		sys.exit(1)

	output_directory = sys.argv[2]

	if os.path.isdir(output_directory):
		eprint("The specified output directory already exists.")
		sys.exit(1)
	elif os.path.isfile(output_directory):
		eprint("The specified output directory is a file.")
		sys.exit(1)
	else:
		try:
			os.mkdir(output_directory)
		except Exception as ex:
			eprint(str(ex))
			eprint("Could not create the output directory.")
			sys.exit(1)

	if output_directory[-1] != '/':
		output_directory += '/'

	return configuration, output_directory

# This copies all of the configuration parameters for the 
# neural network that were specified in the json config file
# into a class that can easily write them into a string for
# export to the final neural network file.
def setup_config_structure(json_config):
	# This class automatically creates a string that can
	# be written directly into a neural network file.
	network_config = TrainingFileConfig()

	# Set all of the members of the configuration structure.
	hyperparameters = json_config["parameter-set"]

	network_config.gi_mode                = hyperparameters["gi_mode"]
	network_config.gi_shift               = hyperparameters["gi_shift"]
	network_config.activation_function    = hyperparameters["activation_function"]
	network_config.cutoff_distance        = hyperparameters["cutoff_distance"]
	network_config.truncation_distance    = hyperparameters["truncation_distance"]
	network_config.gi_sigma               = hyperparameters["gi_sigma"]
	network_config.n_legendre_polynomials = len(hyperparameters["legendre_polynomials"])
	network_config.legendre_orders        = hyperparameters["legendre_polynomials"]
	network_config.n_r0                   = len(hyperparameters["r_0_values"])
	network_config.r0                     = hyperparameters["r_0_values"]
	network_config.n_layers               = len(hyperparameters["network_layers"]) + 1

	proper_layers                         = [network_config.n_r0 * network_config.n_legendre_polynomials]
	proper_layers.extend(hyperparameters["network_layers"])
	network_config.layer_sizes            = proper_layers

	network_config.n_species      = 1
	network_config.element        = 'Si'
	network_config.mass           = 28.0855000
	network_config.randomize      = True
	network_config.max_random     = 0.5
	network_config.BOP_param0     = 1
	network_config.BOP_parameters = [10.78701, 5.23771, 4.04092, 1.365, 0.104528, 0.979074, 0.891061, 0.803526]

	return network_config

def run(cmdline, async=False, wd=None):
	if wd != None:
		process = subprocess.Popen(cmdline.split(' '), stdout=subprocess.PIPE, cwd=wd)
	else:
		process = subprocess.Popen(cmdline.split(' '), stdout=subprocess.PIPE)

	if not async:
		output, error = process.communicate()

		if error != '' and error != None:
			eprint("ERROR: %s"%error)
			sys.exit(1)
	else:
		return process

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


# This will copy pyfit-ff and all of its code into run_dir
# replace all values in Config.py with those in config and
# will run pyfit-ff with the specified parameters. If
# config_file_path is set, Config.py will be loaded from there.
def run_pyfit_with_config(config, params, run_dir, config_file_path=None, async=False):
	if not os.path.isdir(run_dir):
		os.mkdir(run_dir)

	# First, copy pyfit-ff.py and subroutines/ to 
	# the run_dir.

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
	if not async:
		os.chdir(run_dir)
		run("python3 pyfit-ff.py %s"%(params))
		os.chdir(current_dir)
	else:
		return run("python3 pyfit-ff.py %s"%(params), async=True, wd=run_dir)



if __name__ == '__main__':
	config, out_dir = parse_and_validate_args()

	general_log = out_dir + 'garbage.txt'

	# Step One: Generate a neural network file and run pyfit-ff.py to generate
	#           a corresponding LSPARAM file.

	nn_config = setup_config_structure(config)

	# Export the neural network file.
	neural_network_path = out_dir + 'initial-nn-config.dat'
	nn_file = open(neural_network_path, 'w')
	nn_file.write(nn_config.toFileString())
	nn_file.close()

	# Now we run pyfit-ff.py with the new neural network file and
	# instruct it to generate an LSPARAM file.
	lsparam_gen_output_dir = out_dir + 'initial-lsparam-gen/'
	lsparam_gen_config = {
		'NEURAL_NETWORK_FILE'      : '\'%s\''%os.path.abspath(neural_network_path),
		'POSCAR_DATA_FILE'         : '\'%s\''%os.path.abspath(config["poscar_data_file"]),
		'LSPARAM_FILE'             : '\'%s\''%'lsparam-generated.dat'
	}

	# Run the program.
	run_pyfit_with_config(
		lsparam_gen_config,
		'-g -u',
		lsparam_gen_output_dir,
		config_file_path=config["default_config"]
	)

	# Now we have an LSPARAM file ready. We can use the data in it to
	# generate a feature-feature correlation matrix and to export 
	# scatter plot images for human analysis.

	lsparam_path = os.path.abspath(lsparam_gen_output_dir + 'lsparam-generated.dat')

	# Now we run FFCorrelationCalc.py to compute the correlation coefficients
	# and export them, as well as convenient arrays of datapoints for plotting.

	ff_correlation_dir  = out_dir + 'feature-feature-correlation/'
	ff_correlation_file = ff_correlation_dir + 'correlation.json'

	if not os.path.isdir(ff_correlation_dir):
		os.mkdir(ff_correlation_dir)

	# This will generate a correlation file for us.
	run("python3 scripts/FFCorrelationCalc.py %s %s %s"%(
		lsparam_path,
		ff_correlation_file,
		os.path.abspath(general_log)
	))

	# Run FFHeatmap.py to generate a heatmap image file.
	heatmap_path = os.path.abspath(ff_correlation_dir + 'ff-heatmap.png')
	correlations = os.path.abspath(ff_correlation_file)
	resolution   = config["feature_feature_correlation"]["matrix_resolution"]
	abs_string   = 'y' if config["feature_feature_correlation"]["matrix_abs"] else 'n'

	run("python3 scripts/FFHeatmap.py %s %s %s %s"%(
		correlations,
		heatmap_path,
		resolution,
		abs_string
	))

	# Now that we have a heatmap, we want to run FFScatterPlots.py to
	# generate a list of scatterplot png files.

	ratio              = config["feature_feature_correlation"]["scatterplot_ratio"]
	scatter_plots_path = ff_correlation_dir + 'scatterplots/'
	if not os.path.isdir(scatter_plots_path):
		os.mkdir(scatter_plots_path)

	if config["feature_feature_correlation"]["export_scatter"]:
		run("python3 scripts/FFScatterPlots.py %s %s %s"%(
			correlations,
			scatter_plots_path,
			str(ratio)
		))

	# Now we need to generate a number of neural network randomizations
	# specified by the user in the config file and train them to the 
	# number of iterations that they specified. We then use these networks
	# to generate feature-output correlation heatmaps and scatterplots.
	n_networks   = config["feature_output_correlation"]["number_of_networks"]
	n_iterations = config["feature_output_correlation"]["number_of_iterations"]
	n_backups    = config["feature_output_correlation"]["number_of_backups"]
	learn_rate   = config["feature_output_correlation"]["learning_rate"]

	backup_interval = int(np.floor(n_iterations / n_backups))

	training_output_dir     = out_dir + 'trained-networks/'

	if not os.path.isdir(training_output_dir):
		os.mkdir(training_output_dir)

	training_output_subdirs = []
	training_config = {
		'NEURAL_NETWORK_FILE'         : '\'%s\''%os.path.abspath(neural_network_path),
		'POSCAR_DATA_FILE'            : '\'%s\''%os.path.abspath(config["poscar_data_file"]),
		'LSPARAM_FILE'                : '\'%s\''%'lsparam-generated.dat',
		'TRAINING_SET_FILE'           : '\'%s\''%lsparam_path,
		'NETWORK_BACKUP_INTERVAL'     : str(backup_interval),
		'LEARNING_RATE'               : str(learn_rate),
		'MAXIMUM_TRAINING_ITERATIONS' : str(n_iterations)
	}

	processes_to_wait = []

	for initial_condition in range(n_networks):
		this_dir = training_output_dir + 'IC-%02i/'%initial_condition
		bk_dir   = os.path.abspath(this_dir + 'nn_backup/') + '/'

		training_output_subdirs.append(this_dir)

		this_config = copy.deepcopy(training_config)
		this_config["NEURAL_NETWORK_SAVE_FILE"] = '\'%s\''%os.path.abspath(this_dir + 'saved_nn.dat')
		this_config["NETWORK_BACKUP_DIR"]       = '\'%s\''%bk_dir

		if not os.path.isdir(this_dir):
			os.mkdir(this_dir)

		if not os.path.isdir(bk_dir):
			os.mkdir(bk_dir)

		# Run the program.
		proc = run_pyfit_with_config(
			this_config,
			'-t -u -r',
			this_dir,
			config_file_path=config["default_config"],
			async=True
		)

		processes_to_wait.append(proc)

	# By this point, all of the training processes are running.
	# Wait for them to all finish.
	wait_for_processes(processes_to_wait)

	# Now that the training is complete, we need to determine the
	# feature-output correlations and create heatmaps and scatterplots.

	# We need to start with the final network for each initial condition.
	# After that, we should generate more heatmaps using each of the backups.
	# Before we generate any heatmaps, we need to evaluate the neural networks.

	files_to_delete = []

	for initial_condition in range(n_networks):
		this_dir  = training_output_dir + 'IC-%02i/'%initial_condition
		bk_dir    = os.path.abspath(this_dir + 'nn_backup/') + '/'
		nn_path   = os.path.abspath(this_dir + 'saved_nn.dat')
		eval_file = os.path.abspath(this_dir + 'nn_evaluated.json')

		files_to_delete.append(eval_file)

		run("python3 scripts/EvalNN.py %s %s %s"%(
			nn_path,
			lsparam_path,
			eval_file
		))

		# Now we also produce an evaluation for each backup.
		for bk_idx in range(n_backups):
			current_backup = os.path.abspath(bk_dir + 'nn_bk_%i.dat'%bk_idx)
			eval_file      = os.path.abspath(this_dir + 'nn_evaluated_bk_%i.json'%bk_idx)
			run("python3 scripts/EvalNN.py %s %s %s"%(
				current_backup,
				lsparam_path,
				eval_file
			))

			files_to_delete.append(eval_file)


	# Now we have all of the data that we need to generate correlation
	# coefficients, heatmaps and scatterplots.

	# First, generate a correlation file for each neural network and
	# all backups.

	for initial_condition in range(n_networks):
		this_dir  = training_output_dir + 'IC-%02i/'%initial_condition
		bk_dir    = os.path.abspath(this_dir + 'nn_backup/') + '/'
		nn_path   = os.path.abspath(this_dir + 'saved_nn.dat')
		eval_file = os.path.abspath(this_dir + 'nn_evaluated.json')
		correlation_file = os.path.abspath(this_dir + 'correlation.json')

		files_to_delete.append(correlation_file)

		run("python3 scripts/CFCorrelationCalc.py %s %s %s"%(
			eval_file,
			nn_path,
			correlation_file
		))

		# Now we also produce correlation data for each backup.
		for bk_idx in range(n_backups):
			current_backup = os.path.abspath(bk_dir + 'nn_bk_%i.dat'%bk_idx)
			eval_file      = os.path.abspath(this_dir + 'nn_evaluated_bk_%i.json'%bk_idx)
			correlation_file = os.path.abspath(this_dir + 'nn_correlation_bk_%i.json'%bk_idx)

			files_to_delete.append(correlation_file)

			run("python3 scripts/CFCorrelationCalc.py %s %s %s"%(
				eval_file,
				current_backup,
				correlation_file
			))



	# Now that we have all of the correlation data that we need,
	# we invoke the script that creates heatmaps, passing it lists
	# of correlation data that we want compared.

	# Create a list of files.
	# Each list needs to have one file per initial condition.
	# We will create one list for each backup plus the full
	# training file.
	file_sets = []
	for idx in range(n_backups + 1):
		if idx == 0:
			file_name = 'correlation.json'
		else:
			file_name = 'nn_correlation_bk_%i.json'%(idx - 1)

		file_list = []
		for initial_condition in range(n_networks):
			this_dir         = training_output_dir + 'IC-%02i/'%initial_condition
			correlation_file = os.path.abspath(this_dir + file_name)
			file_list.append(correlation_file)

		file_sets.append(file_list)


	abs_string = 'y' if config["feature_output_correlation"]["matrix_abs"] else 'n'

	# Now that we have all sets of files that we want a heatmap for,
	# we invoke CFHeatmap.py for each set of files.
	for idx, fset in enumerate(file_sets):
		png_path = training_output_dir + 'correlation-plots/'

		if not os.path.isdir(png_path):
			os.mkdir(png_path)

		if idx == 0:
			png_path += 'correlation-final.png'
		else:
			png_path += 'correlation-%02i.png'%idx

		full_paths = []
		for path in fset:
			full_paths.append(os.path.abspath(path))

		path_string = ' '.join(full_paths)

		run("python3 scripts/CFHeatmap.py %s %s %s"%(
			png_path,
			abs_string,
			path_string
		))


	scatter_plots_path = os.path.abspath(training_output_dir + 'correlation-scatter-plots/')
	if not os.path.isdir(scatter_plots_path):
		os.mkdir(scatter_plots_path)

	if scatter_plots_path[-1] != '/':
		scatter_plots_path += '/'

	# Now that we have the heatmap written, we need to produce a set
	# of scatterplots for all combinations of parameters and energies.
	# We will only do this for the first initial condition.
	if config["feature_output_correlation"]["export_scatter"]:
		this_dir         = training_output_dir + 'IC-00/'
		correlation_file = os.path.abspath(this_dir + 'correlation.json')

		run("python3 scripts/CFScatterPlots.py %s %s %s"%(
			correlation_file,
			scatter_plots_path,
			'1.0'
		))


	# Now that all graphics are generated, we need to determine the
	# following scoring factors.
	#     1) average feature - feature pcc
	#     2) average feature - output  pcc
	#     3) figure of merit
	#     4) average rmse across networks
	#     5) standard deviation of rmse across networks
	#     6) minimum rmse across networks
	#     7) maximum rmse across networks

	# ff_correlation_file
	# this_dir         = training_output_dir + 'IC-00/'
	# correlation_file = os.path.abspath(this_dir + 'correlation.json')
	# for initial_condition in range(n_networks):
	# 	this_dir  = training_output_dir + 'IC-%02i/'%initial_condition

	# First we load the feature-feature correlation file and
	# compute the average.

	def load_json(file):
		f = open(file, 'r')
		j = json.loads(f.read())
		f.close()
		return j

	ff_correlations  = load_json(ff_correlation_file)
	all_coefficients = np.array([c["pcc"] for c in ff_correlations["coefficients"]])

	mean_ff_correlation = all_coefficients.mean()


	all_rmse         = []
	all_coefficients = []

	for initial_condition in range(n_networks):
		correlation_file = training_output_dir + 'IC-%02i/correlation.json'%initial_condition
		error_file       = training_output_dir + 'IC-%02i/loss_log.txt'%initial_condition

		f   = open(error_file, 'r')
		raw = f.read()
		f.close()
		final_rmse = float([n for n in raw.split('\n') if n != ''][-1])

		all_rmse.append(final_rmse)

		correlation_data = load_json(correlation_file)
		all_coefficients_this_nn = [res['pcc'] for res in correlation_data["data"]]

		all_coefficients.extend(all_coefficients_this_nn)

	mean_fc_correlation = np.array(all_coefficients).mean()
	all_rmse            = np.array(all_rmse)
	mean_rmse           = all_rmse.mean()
	std_rmse            = all_rmse.std()
	min_rmse            = all_rmse.min()
	max_rmse            = all_rmse.max()

	k = len(config["parameter-set"]["r_0_values"]) * len(config["parameter-set"]["legendre_polynomials"])

	figure_of_merit = (k * mean_fc_correlation) / np.sqrt(k + k*(k - 1)*mean_ff_correlation)

	master_results = {}
	master_results["parameter_set"]   = config["parameter-set"]
	master_results["training_params"] = config["feature_output_correlation"]
	master_results["scores"]          = {
		"figure_of_merit"     : figure_of_merit,
		"mean_ff_correlation" : mean_ff_correlation,
		"mean_fc_correlation" : mean_fc_correlation,
		"mean_rmse"           : mean_rmse,
		"std_rmse"            : std_rmse,
		"min_rmse"            : min_rmse,
		"max_rmse"            : max_rmse
	}

	f = open(out_dir + 'master_results.json', 'w')
	f.write(json.dumps(master_results))
	f.close()

	if config["cleanup_large_files"]:
		for file in files_to_delete:
			run("rm %s"%file)
