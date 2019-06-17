import json
import numpy as np
import matplotlib.pyplot
import datetime
import time
import os
import sys
import copy
import subprocess

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

if __name__ == '__main__':
	legendre_sets = [
		[0, 1],
		[0, 1, 2],
		[0, 1, 2, 3],
		[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4, 5],
		[0, 1, 2, 3, 4, 5, 6],
		[0, 1, 2, 3, 4, 5, 6, 7],
		[0, 1, 2, 3, 4, 5, 6, 7, 8]
	]

	sigma_sets = [
		0.1, 0.25, 0.5, 1.0, 1.5, 2
	]

	r0_counts = [
		2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16
	]

	mode_sets = [
		[1, 0.5, 1],
		[0, 0.0, 0],
		[0, 0.0, 1]
	]

	f = open('parameter-set-configuration.json', 'r')
	config = json.loads(f.read())
	f.close()

	configurations = []
	config_dirs    = []
	idx = 0

	for lset in legendre_sets:
		for sigma in sigma_sets:
			for nr0 in r0_counts:
				for mode in mode_sets:
					dir_name = 'eval-data/l_%i_r_%i_s_%f_m_%i_idx_%i'%(
						len(lset),
						nr0,
						sigma,
						mode[0],
						idx
					)
					idx += 1
					config_dirs.append(dir_name + '/')
					new_config = copy.deepcopy(config)

					new_config["parameter-set"]["legendre_polynomials"] = lset
					new_config["parameter-set"]["gi_sigma"]             = sigma
					new_config["parameter-set"]["gi_mode"]              = mode[0]
					new_config["parameter-set"]["gi_shift"]             = mode[1]
					new_config["parameter-set"]["activation_function"]  = mode[2]
					new_config["parameter-set"]["r_0_values"]           = np.linspace(1.8, 4.4, nr0).tolist()
					configurations.append(new_config)

	print("Configurations: %i"%len(configurations))

	config_names = []
	
	for i in range(len(configurations)):
		fname = 'config_%i.json'%i
		
		f = open(fname, 'w')
		f.write(json.dumps(configurations[i]))
		f.close()
		config_names.append(fname)

	for name, _dir in zip(config_names, config_dirs):
		run("./run_eval_enki_generic.sh 4 %s %s"%(name, _dir))

