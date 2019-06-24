import json
import numpy as np
import datetime
import time
import os
import sys
import copy
import subprocess



if __name__ == '__main__':
	# We want to load a list of best indices and then generate a script that 
	# will run all of them on ENKI. We also want to modify some parameters, 
	# primarily the number of training iterations.

	# Load the list of best indices.
	f = open('top_results_01.json', 'r')
	data = json.loads(f.read())
	f.close()

	best_indices = data['best_indices']

	new_config_dir = 'config_rerun/'

	if not os.path.isdir(new_config_dir):
		os.mkdir(new_config_dir)


	# Now we load each of the config files from the list of best configurations
	# and modify them and then copy them to thr output directory.

	for idx in best_indices:
		fname = 'config/config_%05i.json'%idx
		f     = open(fname, 'r')
		cfg   = json.loads(f.read())
		f.close()

		cfg['minimum_output']                                     = False	
		cfg['poscar_data_file']                                   = 'input/Feature_Selection/SET-ADAM-RM-BAD-CLUSTERS-06-14-19-POSCAR-E-full.dat'
		cfg['feature_output_correlation']['number_of_iterations'] = 6000
		cfg['feature_output_correlation']['export_scatter']       = True

		outname = new_config_dir + 'config_%05i.json'%idx
		f = open(outname, 'w')
		f.write(json.dumps(cfg))
		f.close()

	scriptfile = 'rerun.sh'
	f = open(scriptfile, 'w')

	for idx in best_indices:
		outname = new_config_dir + 'config_%05i.json'%idx
		outdir  = '/home/ajr6/2019-06-24/major-sweep-deep-run/idx_%05i/'%idx

		f.write("./run_eval_enki_generic.sh 1 %s %s\n"%(outname, outdir))
		f.write("sleep 0.1\n")

	f.close()
		


