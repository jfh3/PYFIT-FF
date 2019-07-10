import numpy as np
import code
import argparse
import os
import time
import sys
import json

from sys import path
path.append("subroutines")

from TrainingSet import TrainingSetFile
import Util
import subprocess

def unique_combos(l):
	grid         = np.mgrid[0:l, 0:l].swapaxes(0, 2).swapaxes(0, 1)
	m            = grid.shape[0]
	r, c         = np.triu_indices(m, 1)
	combinations = grid[r, c]

	left  = combinations[:,0].astype(np.int32)
	right = combinations[:,1].astype(np.int32)

	return left, right

def run(cmdline, _async=False, wd=None):
	process = subprocess.Popen(
		cmdline.split(' '),
		stdout=subprocess.DEVNULL, 
		stderr=subprocess.DEVNULL
	)

	if not _async:
		output, error = process.communicate()

		if error != '' and error != None:
			eprint("ERROR: %s"%error)
			sys.exit(1)
	else:
		return process


if __name__ == '__main__':
	# This program is meant to generate a list of all unique combinations of
	# two groups and to start a job that calculates the correlations for each
	# pair of groups. 
	#
	# It operates in two modes. 
	#    1) Keep one subprocess running per core, at any given time.
	#    2) Schedule a slurm job for each pair of groups.
	
	parser = argparse.ArgumentParser(
		description="Sets up a job to calculate the cross correlation for " +
		"all pairs of structural groups in a training set file. Can do this " +
		"via slurm jobs or via subprocesses."
	)

	parser.add_argument(
		'-t', '--training-set', dest='tfile', type=str, required=True,
		help='The training set file to read structural parameters from.'
	)

	parser.add_argument(
		'-s', '--slurm-mode', dest='slurm_mode', action='store_true',
		help='Schedule slurm jobs.'
	)

	parser.add_argument(
		'-p', '--subprocess-mode', dest='proc_mode', action='store_true',
		help='Run subprocesses instead of scheduling slurm jobs.'
	)

	parser.add_argument(
		'-n', '--n-procs', dest='n_procs', type=int, default=0,
		help='The number of processes to use if -p/--subprocess-mode is specified'
	)

	parser.add_argument(
		'-b', '--subset', dest='subset', type=int, default=0,
		help='Randomly pick this many groups to do calculations on, instead of all groups.'
	)

	parser.add_argument(
		'-o', '--output-dir', dest='odir', type=str, required=True,
		help='The directory to write the output files to. It will be created if ' +
		'it doesn\'t exist.'
	)

	parser.add_argument(
		'-e', '--slurm-template', dest='slurm_template', type=str, default='',
		help='The template file to use for submitting slurm jobs.'
	)

	args = parser.parse_args(sys.argv[1:])

	if not os.path.isdir(args.odir):
		try:
			# This try-catch is here in case another instance creates the 
			# directory in the time it takes to get from the check to here.
			os.mkdir(args.odir)
		except:
			time.sleep(0.2)
			if not os.path.isdir(args.odir):
				print("Could not create the output directory.")
				exit(1)

	if not args.slurm_mode and not args.proc_mode:
		print("Either -s/--slurm-mode or -p/--subprocess mode must be specified.")
		parser.print_help()
		exit(1)

	if args.slurm_mode:
		print("Slurm mode not implemented.")
		exit(1)

	if args.proc_mode and args.n_procs == 0:
		print("If -p/--subprocess-mode is specified, -n/--n-procs must also be specified.")
		exit(1)

	# Load the training file and build a list of groups.

	Util.init('log.txt')
	training_set = TrainingSetFile(args.tfile)

	all_groups = []
	for struct_id in training_set.training_structures:
		all_groups.append(training_set.training_structures[struct_id][0].group_name)

	all_groups = np.unique(all_groups).tolist()

	# If the user wants only a random subset, get that subset.

	if args.subset != 0:
		all_groups = np.random.choice(all_groups, args.subset, replace=False).tolist()
		print(all_groups)

	# Generate all unique combinations.

	lidx, ridx = unique_combos(len(all_groups))
	names = [(all_groups[l], all_groups[r]) for l, r in zip(lidx, ridx)]

	if args.proc_mode:
		n_complete = 0
		completed  = []
		running    = []

		while n_complete < len(names):
			for idx in range(len(names)):
				if len(running) == args.n_procs:
					break
				if idx not in completed:
					print("Starting %i"%idx)
					cmd = 'python3 structure-lsp-correlation.py -t %s -l %s -r %s -o %s'%(
						args.tfile, names[idx][0], names[idx][1], args.odir
					)

					if names[idx][0] == names[idx][1]:
						print("AKJFALKFJLKDSJLFLKSD")
						exit(1)
					completed.append(idx)
					proc = run(cmd, _async=True)
					running.append(proc)

			remove = None
			for proc in running:
				if proc.poll() is not None:
					print('Process done')
					n_complete += 1
					# This process is done, add its index to the completed list
					# and remove it from the list of those currently running.
					
					remove = proc
					break

			if remove is not None:
				print("Removing from list")
				running.remove(remove)

			time.sleep(0.1)











