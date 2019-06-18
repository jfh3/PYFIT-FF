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
		0.1, 
		0.25,
		0.5, 
		1.0, 
		1.5, 
		2
	]

	r0_counts = [
		2, 
		3, 
		4, 
		5, 
		6, 
		8, 
		9, 
		10, 
		12, 
		14, 
		16
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
					dir_name = '/home/ajr6/2019-06-17/feature-set-02/idx_%i'%(
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

	if not os.path.isdir('config'):
		os.mkdir('config')

	config_names = []
	print(len(configurations))
	for i in range(len(configurations)):
		fname = 'config/config_%i.json'%i
		
		f = open(fname, 'w')
		f.write(json.dumps(configurations[i]))
		f.close()
		config_names.append(fname)

	runfile = open('run_evals.sh', 'w')

	for name, _dir in zip(config_names, config_dirs):
		runfile.write("./run_eval_enki_generic.sh 1 %s %s\n"%(name, _dir))
		runfile.write("sleep 1\n")

	runfile.close()

