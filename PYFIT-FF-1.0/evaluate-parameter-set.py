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
import os
import sys
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

def run(cmdline):
	process = subprocess.Popen(cmdline.split(' '), stdout=subprocess.PIPE)
	output, error = process.communicate()

	if error != '' and error != None:
		eprint("ERROR: %s"%error)
		sys.exit(1)

# This will copy pyfit-ff and all of its code into run_dir
# replace all values in Config.py with those in config and
# will run pyfit-ff with the specified parameters. If
# config_file_path is set, Config.py will be loaded from there.
def run_pyfit_with_config(config, params, run_dir, config_file_path=None):
	if not os.path.isdir(run_dir):
		os.mkdir(output_directory)

	# First, copy pyfit-ff.py and subroutines/ to 
	# the run_dir.

	pyfit_path       = run_dir + "pyfit-ff.py"
	subroutines_path = run_dir + "subroutines/"
	config_path      = run_dir + "subroutines/Config.py"

	if config_file_path != None:
		config_to_load = config_file_path
	else:
		config_to_load = config_path

	if not os.path.isdir(subroutines_path):
		os.mkdir(subroutines_path)

	run("cp -r pyfit-ff.py  %s"%(pyfit_path))
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
	os.chdir(run_dir)
	run("python3 pyfit-ff.py %s"%(params))
	os.chdir(current_dir)

if __name__ == '__main__':
	config, out_dir = parse_and_validate_args()

	# Step One: Generate a neural network file and run pyfit-ff.py to generate
	#           a corresponding LSPARAM file.

	nn_config = setup_config_structure(config)

	# Export the neural network file.
	neural_network_path = out_dir + 'initial-nn-config.dat'
	nn_file = open(neural_network_path, 'w')
	nn_file.write(nn_config.toFileString())
	nn_file.close()

	# Now we run pyfit-ff.py with the new neural network file and
	# instruct it to generate and LSPARAM file.