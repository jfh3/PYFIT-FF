import json
import numpy as np
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
	# legendre_sets = [
	# 	[0, 1],
	# 	[0, 1, 2],
	# 	[0, 1, 2, 3],
	# 	[0, 1, 2, 3, 4],
	# 	[0, 1, 2, 3, 4, 5],
	# 	[0, 1, 2, 3, 4, 5, 6],
	# 	[0, 1, 2, 3, 4, 5, 6, 7],
	# 	[0, 1, 2, 3, 4, 5, 6, 7, 8]
	# ]

	# r0_counts = [
	# 	2, 
	# 	3, 
	# 	4, 
	# 	5, 
	# 	6, 
	# 	8, 
	# 	9, 
	# 	10, 
	# 	12, 
	# 	14, 
	# 	16
	# ]

	# valid_n_legendre    = range(4, 9)
	# valid_legendre_poly = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	# valid_n_r0          = range(2, 15)
	# valid_r0_range      = [1.0, 5.0]

	# legendre_sets = []
	# r0_sets       = []
	# for i in range(13):
	# 	n_leg = np.random.choice(valid_n_legendre, 1)
	# 	legendre_sets.append(sorted(np.random.choice(valid_legendre_poly, n_leg, replace=False).tolist()))

	# 	n_r0    = np.random.choice(valid_n_r0, 1)
	# 	r0_rand = np.random.uniform(valid_r0_range[0], valid_r0_range[1], 128)
	# 	threshold = ((valid_r0_range[1] - valid_r0_range[0]) / n_r0) / 3

	# 	while True:
	# 		r0_values = sorted(np.random.choice(r0_rand, n_r0, replace=False).tolist())
	# 		# Make sure that none of them are too close.
	# 		min_dist = 1000.0
	# 		for idx, val in enumerate(r0_values):
	# 			for idx_inner, val_inner in enumerate(r0_values):
	# 				if idx_inner != idx:
	# 					dist = np.abs(val - val_inner)
	# 					if dist < min_dist:
	# 						min_dist = dist
	# 		if min_dist > threshold:
	# 			r0_sets.append(r0_values)
	# 			break
	# 		else:
	# 			print("Regenerating list due to close together r0 values.")

	# r0_sets.append([2.33, 3.66])

	r0_sets       = []
	legendre_sets = []

	valid_legendre_poly = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	valid_r0_range      = [2.0, 4.0]


	r0_sets.append([3.0])
	r0_sets.append([2.5, 3.5])
	r0_sets.append([2.5, 3.0, 3.5])
	for i in range(4, 21):
		r0_sets.append(np.linspace(2.0, 4.0, i).tolist())

	for i in range(30):
		n = int(round(np.random.uniform(1, 16)))
		r0_sets.append(np.random.normal(3.0, 0.66, n).tolist())

	for i in range(80):
		n    = int(round(np.random.uniform(2, 10)))
		set_ = np.random.choice(valid_legendre_poly, n, replace=False)
		set_.sort()
		legendre_sets.append(set_.tolist())

	
	# Select 13 sets of legendre polynomials
	# and 13 sets of r0 values.

	sigma_sets = [ 
		0.5,
		1.0,
		1.5, 
		2.0
	]

	

	mode_sets = [
		[1, 0.5, 1],
		
	]

	network_structures = [
		[16, 16, 1],
	]

	f = open('parameter-set-configuration.json', 'r')
	config = json.loads(f.read())
	f.close()

	configurations = []
	config_dirs    = []
	idx = 0

	for network in network_structures:
		for lset in legendre_sets:
			for sigma in sigma_sets:
				for r0 in r0_sets:
					for mode in mode_sets:
						dir_name = '/home/ajr6/2019-06-21/major-sweep-test-01/idx_%05i'%(
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
						new_config["parameter-set"]["r_0_values"]           = r0
						new_config["parameter-set"]["network_layers"]       = network
						configurations.append(new_config)

	print("Configurations: %i"%len(configurations))

	if not os.path.isdir('config'):
		os.mkdir('config')

	config_names = []
	for i in range(len(configurations)):
		fname = 'config/config_%05i.json'%i
		
		f = open(fname, 'w')
		f.write(json.dumps(configurations[i]))
		f.close()
		config_names.append(fname)

	runfile = open('run_evals.sh', 'w')

	for name, _dir in zip(config_names, config_dirs):
		runfile.write("./run_eval_enki_generic.sh 1 %s %s\n"%(name, _dir))
		runfile.write("sleep 0.1\n")

	runfile.close()

