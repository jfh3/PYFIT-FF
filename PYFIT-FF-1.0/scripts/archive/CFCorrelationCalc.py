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
from NeuralNetwork       import NeuralNetwork
import Util

# Print to stderr.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def CFCorrelationCalc(input_data, eval_data, nn_config_file):
	Util.init("garbage.txt")
	Util.set_mode(unsupervised=True)

	nn_config = NeuralNetwork(nn_config_file).config
	legendre_polynomials = nn_config.legendre_orders
	r_0_values           = nn_config.r0
	n_atoms              = len(eval_data['output'])

	results = {"data" : []}

	energy       = np.array(eval_data['output'])
	energy_mean  = energy.mean()
	energy_std   = energy.std()
	energy_diffs = energy - energy_mean

	for param in range(nn_config.n_r0 * nn_config.n_legendre_polynomials):
		# This is added into results["coefficients"] at the end.
		current_result = {
			'param' : {
				'idx' : param,
				'r0'  : r_0_values[param % len(r_0_values)],
				'l'   : legendre_polynomials[param // len(r_0_values)]
			}
		}

		inputs     = input_data[param]
		input_mean = inputs.mean()
		
		diff        = inputs - input_mean
		numerator   = (energy_diffs * diff).mean()
		denominator = energy_std * inputs.std()

		# We want export the data points that were used for this
		# process so that the next script can generate scatterplots.

		current_result['pcc'] = numerator / denominator
		results["data"].append(current_result)

	Util.cleanup()

	return results

# if __name__ == '__main__':
# 	# This takes three parameters:
# 	#     1) The neural network evaluation data file.
# 	#     2) The neural network file to get parameters from.
# 	#     3) The file to write results to.

# 	if len(sys.argv) != 4:
# 		eprint("This program takes 3 arguments.")
# 		sys.exit(1)

	
# 	nn_eval_file   = sys.argv[1]
# 	nn_config_file = sys.argv[2]
# 	output_file    = sys.argv[3]

	

# 	f = open(nn_eval_file, 'r')
# 	eval_data = json.loads(f.read())
# 	f.close()

# 	# Load whichever training set is specified above.

# 	results = CFCorrelationCalc(eval_data, nn_config_file)

	

# 	f = open(output_file, 'w')
# 	f.write(json.dumps(results))
# 	f.close()