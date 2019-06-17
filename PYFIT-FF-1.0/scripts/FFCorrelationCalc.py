import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import json
from   time     import time
from   datetime import datetime

from sys import path
path.append("subroutines")

from TrainingSet         import TrainingSetFile
from ConfigurationParser import TrainingFileConfig
import Util

# Print to stderr.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def FFCorrelationCalc(training_set_file, log_file):
	Util.init(log_file)
	Util.set_mode(unsupervised=True)

	# Load whichever training set is specified above.
	training_set      = TrainingSetFile(training_set_file)
	n_params_per_atom = len(training_set.training_structures[0][0].structure_params)
	n_atoms           = training_set.n_atoms
	n_coefficients    = (n_params_per_atom**2 - n_params_per_atom) // 2

	print("Calculating Pearson Correlation Coefficients")
	print("\tNumber of Atoms                     = {:,}".format(n_atoms))
	print("\tNumber of Structures                = {:,}".format(training_set.n_structures))
	print("\tNumber of Parameters Per Atom       = {:,}".format(n_params_per_atom))
	print("\tNumber of Parameters (Total)        = {:,}".format(n_params_per_atom*n_atoms))
	print("\tNumber of Coefficients to Calculate = {:,}".format(n_coefficients))

	results = {}

	# We want to store these values for later so we can write them
	# to the output JSON file. This should ensure that all useful
	# information about this process is preserved in the file.
	legendre_polynomials = training_set.config.legendre_orders
	r_0_values           = training_set.config.r0
	cutoff               = training_set.config.cutoff_distance
	truncation           = training_set.config.truncation_distance
	sigma                = training_set.config.gi_sigma
	mode                 = training_set.config.gi_mode
	shift                = training_set.config.gi_shift
	start                = str(datetime.now())

	# Start setting up the results structure.
	results["training_set_file"]    = training_set_file
	results["legendre_polynomials"] = legendre_polynomials
	results["r_0_values"]           = r_0_values
	results["cutoff"]               = cutoff
	results["truncation"]           = truncation
	results["sigma"]                = sigma
	results["mode"]                 = mode
	results["shift"]                = shift
	results["start"]                = start
	results["n_atoms"]              = n_atoms
	results["n_params_per_atom"]    = n_params_per_atom
	results["coefficients"]         = []

	# Pull all of the structure parameters out of it and make
	# them into Ng arrays of Na values where Ng is the number
	# of Gi's per atom and Na is the number of atoms.
	all_params = np.zeros((n_params_per_atom, n_atoms))

	# We need to flatten the training structure into a 
	# list of atoms, not subdivided by structure.

	all_atoms = []
	for struct_id in training_set.training_structures.keys():
		struct       = training_set.training_structures[struct_id]
		param_arrays = [atom.structure_params for atom in struct]

		all_atoms.extend(param_arrays)

	# Now that we have a list of all atoms, form the array of
	# structure parameters. To anyone looking at this, yes
	# it is basically just a transpose operation.
	for param_idx in range(n_params_per_atom):
		for atom_idx in range(n_atoms):
			all_params[param_idx][atom_idx] = all_atoms[atom_idx][param_idx]

	# Now we have an array that is convenient for these calculations.
	# We need to generate all unique combinations of two parameters,
	# commutatively, and calculate the Pearson Correlation Coefficient
	# for them. This will also necessitate the use of the standard deviation
	# and the mean of each, which are computed below.

	std_vals  = np.zeros(n_params_per_atom)
	mean_vals = np.zeros(n_params_per_atom)
	for idx in range(n_params_per_atom):
		std_vals[idx]  = all_params[idx].std()
		mean_vals[idx] = all_params[idx].mean()


	# Now we generate all unique combination, and for each of them
	# we calculate the pearson correlation coefficient. Don't ask me
	# how this works, I don't really understand it. I have tested it
	# to ensure that it does what is expected though.
	grid         = np.mgrid[0:n_params_per_atom, 0:n_params_per_atom].swapaxes(0, 2).swapaxes(0, 1)
	m            = grid.shape[0]
	r, c         = np.triu_indices(m, 1)
	combinations = grid[r, c]

	left  = combinations[:,0].astype(np.int32)
	right = combinations[:,1].astype(np.int32)

	# Now that we have all of the means, all of the standard deviations and
	# all of the indices, we can perform the actual correlation coefficient
	# calculations.

	bar = Util.ProgressBar("Corellation Coefficients", 30, n_coefficients, update_every = 1)

	n_processed = 1

	for l, r in zip(left, right):
		# This is added into results["coefficients"] at the end.
		current_result = {
			'param0' : {
				'idx' : int(l),
				'r0'  : r_0_values[l % len(r_0_values)],
				'l'   : legendre_polynomials[l // len(r_0_values)]
			},
			'param1' : {
				'idx' : int(r),
				'r0'  : r_0_values[r % len(r_0_values)],
				'l'   : legendre_polynomials[r // len(r_0_values)]
			}
		}

		left_diff   = all_params[l] - mean_vals[l]
		right_diff  = all_params[r] - mean_vals[r]
		numerator   = (left_diff * right_diff).mean()
		denominator = std_vals[l] * std_vals[r]

		# We want export the data points that were used for this
		# process so that the next script can generate scatterplots.

		current_result["data"] = [[first, second] for first, second in zip(all_params[l], all_params[r])]

		current_result['pcc'] = numerator / denominator
		results["coefficients"].append(current_result)

		bar.update(n_processed)
		n_processed += 1

	bar.finish()

	Util.cleanup()

	return results

if __name__ == '__main__':
	# This takes three parameters:
	#     1) The lsparam file to use for calculations.
	#     2) The file to write the resulting correlations into.
	#     3) The log file to use.

	if len(sys.argv) != 4:
		eprint("This program takes 3 arguments.")
		sys.exit(1)

	training_set_file = sys.argv[1]
	results_file      = sys.argv[2]
	log_file          = sys.argv[3]

	results = FFCorrelationCalc(training_set_file, log_file)

	f = open(results_file, 'w')
	f.write(json.dumps(results))
	f.close()