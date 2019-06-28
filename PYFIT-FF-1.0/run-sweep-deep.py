import json
import numpy as np
import datetime
import time
import os
import sys
import copy
import subprocess

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

if __name__ == '__main__':

	def network_params(layers):
		n_params = layers[0] * layers[1] + layers[1]
		for i, l in enumerate(layers[1:-1]):
			n_params += l * layers[i + 2] + layers[i + 2]

		return n_params

	def constrain_network(proposed, desired_params, threshold):
		new_net = copy.deepcopy(proposed)

		def check():
			return np.abs(network_params(new_net) - desired_params) > threshold

		def delta():
			return network_params(new_net) - desired_params

		current_move = 1
		total_iter   = 0
		while check() and total_iter < 128:

			if delta() > threshold:
				indices = list(range(1, len(new_net) - 1))
				indices.reverse()

				for idx in indices:
					new_net[idx] -= 1

					if not check():
						break

			elif delta() < threshold:
				indices = list(range(1, len(new_net) - 1))

				for idx in indices:
					new_net[idx] += 1

					if not check():
						break

			current_move += 1

			total_iter += 1

		def sequential(l):
			for idx, i in enumerate(l):
				if idx != 0:
					if i - l[idx - 1] > 0:
						return False
			return True


		if check():
			# We exceeded the maximum number of logical operations.
			# Time to randomize it.
			while check():
				while not sequential(new_net[1:-1]) or check():
					r_idx = np.random.randint(1, len(new_net) - 1)
					r     = np.random.randint(2, 100)
					new_net[r_idx] = r
					

		if not sequential(new_net[1:-1]):
			raise Exception("WUT")

		return new_net


	n_desired = 1890
	threshold = 21

	r0_sets       = []
	legendre_sets = []

	sigma_sets   = [0.5, 1.0, 2.0]
	base_network = [32, 32, 1]
	n_desired = 1890
	threshold = 21

	r0_sets.append([2.08])
	r0_sets.append([2.48])
	r0_sets.append([2.5, 3.5])
	r0_sets.append([1.58, 2.08, 2.58])
	r0_sets.append([1.98, 2.48, 2.98])
	for i in [4, 6, 8, 10, 12, 16]:
		r0_sets.append(np.linspace(2.0, 4.0, i).tolist())


	legendre_sets.append([0, 1])
	legendre_sets.append([0, 1, 2])
	legendre_sets.append([0, 1, 2, 3])
	legendre_sets.append([0, 1, 2, 3, 4])
	legendre_sets.append([0, 1, 2, 3, 4, 5])
	legendre_sets.append([0, 1, 2, 3, 4, 5, 6])

	legendre_sets.append([0, 1, 2, 4])
	legendre_sets.append([0, 1, 2, 4, 6])

	f = open('parameter-set-configuration.json', 'r')
	config = json.loads(f.read())
	f.close()

	configurations = []
	config_dirs    = []
	idx = 0

	for lset in legendre_sets:
		for sigma in sigma_sets:
			for r0 in r0_sets:
				dir_name = '/home/ajr6/2019-06-28/deep-runs-final/idx_%05i'%(
					idx
				)
				idx += 1
				config_dirs.append(dir_name + '/')
				new_config = copy.deepcopy(config)

				new_config["parameter-set"]["legendre_polynomials"] = lset
				new_config["parameter-set"]["gi_sigma"]             = sigma
				new_config["parameter-set"]["gi_mode"]              = 1
				new_config["parameter-set"]["gi_shift"]             = 0.5
				new_config["parameter-set"]["activation_function"]  = 1
				new_config["parameter-set"]["r_0_values"]           = r0

				n_args = len(r0) * len(lset)
				network = constrain_network([n_args, *base_network], n_desired, threshold)

				new_config["parameter-set"]["network_layers"] = network
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

