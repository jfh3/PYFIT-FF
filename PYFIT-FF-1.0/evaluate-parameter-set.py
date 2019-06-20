# This program takes a file as the first argument, the configuration 
# parameters json file. It's second argument is the folder to create and store 
# the output information in. It is expected to not exist at the time that the 
# program is called. It will generate neural network files, LSParam files, 
# feature-feature correlations, heatmaps, scatterplot pngs, trained neural 
# networks, feature-output correlations, more scatterplot pngs and more 
# heatmaps, as well as actually calculating a final figure of merit score.

import matplotlib
matplotlib.use('Agg')

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

# Print to stderr.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Reads and ensures the validity of the arguments.
# Returns a configuration dictionary and the output
# directory, ensuring that the output directory is
# empty and exists.
#
# return configuration, work_directory, final_directory
def parse_and_validate_args():
	if len(sys.argv) != 4:
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

	work_directory = sys.argv[2]

	if os.path.isfile(work_directory):
		eprint("The specified work directory is a file.")
		sys.exit(1)
	else:
		try:
			if not os.path.isdir(work_directory):
				os.mkdir(work_directory)
		except Exception as ex:
			eprint(str(ex))
			eprint("Could not create the work directory.")
			sys.exit(1)

	if work_directory[-1] != '/':
		work_directory += '/'

	final_directory = sys.argv[3]

	if os.path.isdir(final_directory):
		eprint("The specified output directory already exists.")
		sys.exit(1)
	elif os.path.isfile(final_directory):
		eprint("The specified output directory is a file.")
		sys.exit(1)
	else:
		try:
			os.mkdir(final_directory)
		except Exception as ex:
			eprint(str(ex))
			eprint("Could not create the output directory.")
			sys.exit(1)

	return configuration, work_directory, final_directory

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


	# --------------------------------------------------
	# Setup
	# --------------------------------------------------

	config, wrk_dir, final_dir = parse_and_validate_args()

	subroutines_dirs = []

	general_log = wrk_dir + 'garbage.txt'
	final_data  = {}

	# Step One: Generate a neural network file and run pyfit-ff.py to generate
	#           a corresponding LSPARAM file.

	nn_config = setup_config_structure(config)

	# Export the neural network file.
	neural_network_path = wrk_dir + 'initial-nn-config.dat'
	nn_file = open(neural_network_path, 'w')
	nn_file.write(nn_config.toFileString())
	nn_file.close()

	# --------------------------------------------------
	# Training Set Generation
	# --------------------------------------------------

	# Now we run pyfit-ff.py with the new neural network file and
	# instruct it to generate an LSPARAM file.
	lsparam_gen_output_dir = wrk_dir + 'initial-lsparam-gen/'
	lsparam_gen_config = {
		'NEURAL_NETWORK_FILE'      : '\'%s\''%os.path.abspath(neural_network_path),
		'POSCAR_DATA_FILE'         : '\'%s\''%os.path.abspath(config["poscar_data_file"]),
		'LSPARAM_FILE'             : '\'%s\''%'lsparam-generated.dat'
	}

	lsparam_path = os.path.abspath(lsparam_gen_output_dir + 'lsparam-generated.dat')

	if 'lsparam_file' in config:
		if not os.path.isdir(lsparam_gen_output_dir):
			os.mkdir(lsparam_gen_output_dir)
		run("cp %s %s"%(config['lsparam_file'], lsparam_path))
	else:
		subroutines_dirs.append(lsparam_gen_output_dir + 'subroutines/')
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

	

	# Now we run FFCorrelationCalc.py to compute the correlation coefficients
	# and export them, as well as convenient arrays of datapoints for plotting.

	# --------------------------------------------------
	# Feature Feature Correlation
	# --------------------------------------------------

	ff_correlation_dir  = wrk_dir + 'feature-feature-correlation/'
	ff_correlation_file = ff_correlation_dir + 'correlation.json'

	if not os.path.isdir(ff_correlation_dir):
		os.mkdir(ff_correlation_dir)

	print("START FFCorrelationCalc")
	ff_correlation_data, all_params = FFCorrelationCalc(lsparam_path, general_log)
	print("END   FFCorrelationCalc")

	# We want to take param0, param1 and pcc from this structure and save it for a final
	# output data file.

	final_data['feature-feature-correlations'] = []
	for pcc in ff_correlation_data["coefficients"]:
		final_data['feature-feature-correlations'].append({
			'parameters'  : [pcc['param0'], pcc['param1']],
			'coefficient' :	pcc['pcc']
		})


	# Run FFHeatmap.py to generate a heatmap image file.
	heatmap_path = os.path.abspath(ff_correlation_dir + 'ff-heatmap.png')
	correlations = os.path.abspath(ff_correlation_file)
	resolution   = config["feature_feature_correlation"]["matrix_resolution"]

	GenHeatmap(ff_correlation_data, heatmap_path, config["feature_feature_correlation"]["matrix_abs"])

	# Now that we have a heatmap, we want to run FFScatterPlots.py to
	# generate a list of scatterplot png files.

	ratio              = config["feature_feature_correlation"]["scatterplot_ratio"]
	scatter_plots_path = ff_correlation_dir + 'scatterplots/'
	if not os.path.isdir(scatter_plots_path):
		os.mkdir(scatter_plots_path)

	# TODO: Put this in a separate process.
	# if config["feature_feature_correlation"]["export_scatter"]:
	# 	GenFFScatterPlots(ff_correlation_data, scatter_plots_path, ratio)


	# Now we need to generate a number of neural network randomizations
	# specified by the user in the config file and train them to the 
	# number of iterations that they specified. We then use these networks
	# to generate feature-output correlation heatmaps and scatterplots.
	#n_networks   = config["feature_output_correlation"]["number_of_networks"]
	n_networks   = config["feature_output_correlation"]["number_of_networks"]
	n_iterations = config["feature_output_correlation"]["number_of_iterations"]
	n_backups    = config["feature_output_correlation"]["number_of_backups"]
	learn_rate   = config["feature_output_correlation"]["learning_rate"]

	backup_interval = int(np.round(n_iterations / n_backups))

	training_output_dir     = wrk_dir + 'trained-networks/'

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

	gpu_idx = 0

	
	to_wait = []
	for initial_condition in range(n_networks):
		this_dir = training_output_dir + 'IC-%02i/'%initial_condition
		bk_dir   = os.path.abspath(this_dir + 'nn_backup/') + '/'

		training_output_subdirs.append(this_dir)

		this_config = copy.deepcopy(training_config)
		this_config["NEURAL_NETWORK_SAVE_FILE"]    = '\'%s\''%os.path.abspath(this_dir + 'saved_nn.dat')
		this_config["NETWORK_BACKUP_DIR"]          = '\'%s\''%bk_dir


		if not os.path.isdir(this_dir):
			os.mkdir(this_dir)

		if not os.path.isdir(bk_dir):
			os.mkdir(bk_dir)


		subroutines_dirs.append(this_dir + 'subroutines/')
		print("START Training")
		if config['feature_output_correlation']['train_sequential']:
			run_pyfit_with_config(
				this_config,
				'-t -u -r',
				this_dir,
				config_file_path=config["default_config"],
				async=False
			)
		else:
			# Run the program.
			p = run_pyfit_with_config(
				this_config,
				'-t -u -r --gpu-affinity %i'%gpu_idx,
				this_dir,
				config_file_path=config["default_config"],
				async=True
			)
			gpu_idx += 1
			to_wait.append(p)

	if not config['feature_output_correlation']['train_sequential']:
		wait_for_processes(to_wait)
	print("END  Training")
	# Now that the training is complete, we need to determine the
	# feature-output correlations and create heatmaps and scatterplots.

	# We need to start with the final network for each initial condition.
	# After that, we should generate more heatmaps using each of the backups.
	# Before we generate any heatmaps, we need to evaluate the neural networks.

	print("START load training set")
	training_set     = GetTrainingSetInstance(lsparam_path)
	print("END load training set")
	correlation_sets = [] 

	first_nn_results = None
	for initial_condition in range(n_networks):
		network_correlations = []

		this_dir  = training_output_dir + 'IC-%02i/'%initial_condition
		bk_dir    = os.path.abspath(this_dir + 'nn_backup/') + '/'
		nn_path   = os.path.abspath(this_dir + 'saved_nn.dat')
		eval_file = os.path.abspath(this_dir + 'nn_evaluated.json')

		eval_data             = RunNetwork(nn_path, training_set, all_params)
		correlation_data_main = CFCorrelationCalc(all_params, eval_data, nn_path)

		if initial_condition == 0:
			first_nn_results = eval_data
		

		# Now we also produce an evaluation for each backup.
		for bk_idx in range(n_backups):
			current_backup = os.path.abspath(bk_dir + 'nn_bk_%i.dat'%bk_idx)
			eval_file      = os.path.abspath(this_dir + 'nn_evaluated_bk_%i.json'%bk_idx)
			
			eval_data        = RunNetwork(current_backup, training_set, all_params)
			correlation_data = CFCorrelationCalc(all_params, eval_data, current_backup)
			network_correlations.append(correlation_data)

		network_correlations.append(correlation_data_main)
		correlation_sets.append(network_correlations)



	feature_output_convergence_plot = training_output_dir + 'feature-output-convergence.png'
	
	in_network_convergence_data  = []
	network_converged_indicators = []

	for tmp_correlation in correlation_sets:
		n_params   = len(tmp_correlation[0]['data'])
		param_sets = []
		for param in range(n_params):
			param_data = []
			for step in tmp_correlation:
				param_data.append(step['data'][param]['pcc'])
			param_sets.append(param_data)

		fig, axes = plt.subplots(1, 1)
		for set_ in param_sets:
			axes.plot(range(len(tmp_correlation)), set_)

		x_ticks = range(len(tmp_correlation))[::4]

		axes.set_xlabel('Training Iteration')
		axes.set_xticks(x_ticks)
		axes.set_xticklabels(np.array(x_ticks) * backup_interval)
		axes.set_ylabel('Feature - Output Correlation')
		axes.set_title("Feature Output Correlation Convergence")
		plt.savefig(feature_output_convergence_plot, dpi=250)

		# Here we verify the convergence of the feature-output correlations.
		last_n_to_verify = config['feature_output_correlation']['verify_last_n_convergence']
		std_threshold    = config['feature_output_correlation']['std_threshold']

		parameter_convergences   = []
		all_params_converged     = True

		# Get the last last_n_to_verify elements of each set of coefficients.
		end_coefficient_sets = [a[-last_n_to_verify:] for a in param_sets]
		standard_deviations  = [np.array(a).std() for a in end_coefficient_sets]

		for idx, std in enumerate(standard_deviations):
			parameter_convergences.append({'idx': idx, 'std': std})
			if std > std_threshold:
				all_params_converged = False

		in_network_convergence_data.append(parameter_convergences)
		network_converged_indicators.append(all_params_converged)

	# Now that we have verified that the network converged, we need to make
	# sure that the final value of the correlation is consistent between all
	# networks.
	convergence_threshold                = config['feature_output_correlation']['cross_network_convergence_threshold']
	final_coefficient_values_per_network = []
	for tmp_correlation in correlation_sets:

		n_params   = len(tmp_correlation[0]['data'])
		param_sets = []
		for param in range(n_params):
			param_data = []
			for step in tmp_correlation:
				param_data.append(step['data'][param]['pcc'])
			param_sets.append(param_data)

		# Get the last last_n_to_verify elements of each set of coefficients.
		final_coefficient_values = [a[-1] for a in param_sets]
		final_coefficient_values_per_network.append(final_coefficient_values)
		
	between_network_convergences      = []
	params_converged_between_networks = True
	for idx in range(len(final_coefficient_values_per_network[0])):
		this_param_values = [a[idx] for a in final_coefficient_values_per_network]
		std = np.array(this_param_values).std()
		between_network_convergences.append({'idx': idx, 'std': std})
		if std > convergence_threshold:
			params_converged_between_networks = False


	# We want to store Feature Classification Correlations in the final file as well.
	final_data['feature-output-correlations'] = {}
	for idx, network in enumerate(correlation_sets):
		network_corr = {'final': None, 'backups': {}}
		for idx_bk, stage in enumerate(network):
			data = []
			for parameter in stage['data']:
				data.append({
					'parameter'   : parameter['param'],
					'coefficient' : parameter['pcc']
				})

			if idx_bk != len(network) - 1:
				network_corr['backups']["backup-%03i"%idx_bk] = data
			else:
				network_corr['final'] = data

		final_data['feature-output-correlations']['network-%03i'%idx] = network_corr

	# Now that we have all of the correlation data that we need,
	# we invoke the script that creates heatmaps, passing it lists
	# of correlation data that we want compared.

	# Create a list of files.
	# Each list needs to have one file per initial condition.
	# We will create one list for each backup plus the full
	# training file.


	abs_display = config["feature_output_correlation"]["matrix_abs"]

	# Now that we have all sets of files that we want a heatmap for,
	# we invoke CFHeatmap.py for each set of files.
	for idx in range(n_backups + 1):
		png_path = training_output_dir + 'correlation-plots/'

		if not os.path.isdir(png_path):
			os.mkdir(png_path)

		if idx == n_backups:
			png_path += 'correlation-final.png'
			GenCFHeatmap([c[idx] for c in correlation_sets], abs_display, png_path)
		else:
			png_path += 'correlation-%02i.png'%idx

		



	scatter_plots_path = os.path.abspath(training_output_dir + 'correlation-scatter-plots/')
	if not os.path.isdir(scatter_plots_path):
		os.mkdir(scatter_plots_path)

	if scatter_plots_path[-1] != '/':
		scatter_plots_path += '/'

	# Now that we have the heatmap written, we need to produce a set
	# of scatterplots for all combinations of parameters and energies.
	# We will only do this for the first initial condition.
	if config["feature_output_correlation"]["export_scatter"]:
		final_network_correlations = correlation_sets[0][-1]
		GenCFScatterPlots(all_params, final_network_correlations, first_nn_results["output"], scatter_plots_path)



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

	ff_correlations  = ff_correlation_data
	all_coefficients = np.array([c["pcc"] for c in ff_correlations["coefficients"]])

	mean_ff_correlation = np.abs(all_coefficients).mean()


	all_rmse         = []
	all_coefficients = []

	for initial_condition in range(n_networks):
		correlation_data = correlation_sets[initial_condition][-1]
		error_file       = training_output_dir + 'IC-%02i/loss_log.txt'%initial_condition

		f   = open(error_file, 'r')
		raw = f.read()
		f.close()
		final_rmse = float([n for n in raw.split('\n') if n != ''][-1])

		all_rmse.append(final_rmse)

		all_coefficients_this_nn = [res['pcc'] for res in correlation_data["data"]]

		all_coefficients.extend(all_coefficients_this_nn)

	mean_fc_correlation = np.abs(np.array(all_coefficients)).mean()
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

	master_results['all_params_converged'] = all_params_converged
	master_results['all_params_converged_between_networks'] = params_converged_between_networks

	f = open(wrk_dir + 'master_results.json', 'w')
	f.write(json.dumps(master_results))
	f.close()


	final_data['between_network_convergences'] = between_network_convergences
	final_data['parameter_convergences'] = parameter_convergences
	f = open(wrk_dir + 'final_data.json', 'w')
	f.write(json.dumps(final_data))
	f.close()

	run("rm %s"%(lsparam_path))
	
	for sub in subroutines_dirs:
		try:
			run("rm -rf %s"%sub)
		except:
			print("Failed to remove subroutines directory.")
			print("Please remove manually.")
			break

	run("cp -r %s %s"%(wrk_dir, final_dir))