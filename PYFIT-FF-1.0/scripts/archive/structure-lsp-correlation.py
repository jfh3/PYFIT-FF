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

def unique_combos(l):
	grid         = np.mgrid[0:l, 0:l].swapaxes(0, 2).swapaxes(0, 1)
	m            = grid.shape[0]
	r, c         = np.triu_indices(m, 1)
	combinations = grid[r, c]

	left  = combinations[:,0].astype(np.int32)
	right = combinations[:,1].astype(np.int32)

	return left, right

def construct_self_correlation_arrays(data, params_per_atom):
	l, r = unique_combos(len(data))

	# The two arrays produced by this operation should line up
	# the gi's for the left and the right atom, once for each
	# combo.

	left_result  = np.zeros(l.shape[0] * params_per_atom)
	right_result = np.zeros(r.shape[0] * params_per_atom)

	# Fill the two arrays.
	for idx, (li, ri) in enumerate(zip(l, r)):
		base   = params_per_atom * idx
		stride = base + params_per_atom
		left_result[base:stride]  = data[li]
		right_result[base:stride] = data[ri]

	return left_result, right_result

def pearson(X, Y):
	Xmean = X.mean()
	Ymean = Y.mean()

	num = ((X - Xmean) * (Y - Ymean)).mean()
	den = X.std() * Y.std()

	return num / den

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="When given a training set file and a list of structural " +
		"groups, will determine the cross correlation between the structural " +
		"parameters in both groups.",
		epilog="This program is designed to be run multiple times, with a " +
		"controlling program dispatching parts of a large dataset " +
		"multiple runs of this program."
	)

	parser.add_argument(
		'-t', '--training-set', dest='tfile', type=str, required=True,
		help='The training set file to read structural parameters from.'
	)

	parser.add_argument(
		'-l', '--left-groups', dest='lgroups', type=str, required=True, nargs='*',
		help='The groups that form one half of the pairs to correlate.'
	)

	parser.add_argument(
		'-r', '--right-groups', dest='rgroups', type=str, required=True, nargs='*',
		help='The groups that form the other half of the pairs to correlate.'
	)

	parser.add_argument(
		'-o', '--output-dir', dest='odir', type=str, required=True,
		help='The directory to write the output files to. It will be created if ' +
		'it doesn\'t exist.'
	)

	parser.add_argument(
		'-m', '--max-atoms', dest='max_atoms', type=int, default=0,
		help='The maximum number of atoms to process for a group.'
	)

	args = parser.parse_args(sys.argv[1:])

	# Make sure the output directory exists.
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

	if args.odir[-1] != '/':
		args.odir += '/'

	Util.init('log.txt')
	training_set = TrainingSetFile(args.tfile)

	# Build a list of all of the gi's for every group that was specified.
	# This will be turned into numpy arrays ready for processing in the
	# next step.
	params_per_atom  = 0
	left_group_data  = {}
	right_group_data = {}
	for struct_id in training_set.training_structures:
		struct = training_set.training_structures[struct_id]
		name   = struct[0].group_name

		structure_params = [atom.structure_params for atom in struct]
		params_per_atom  = len(structure_params[0])

		if name in args.lgroups:
			if name in left_group_data:
				left_group_data[name].extend(structure_params)
			else:
				left_group_data[name] = structure_params
		
		if name in args.rgroups:
			if name in right_group_data:
				right_group_data[name].extend(structure_params)
			else:
				right_group_data[name] = structure_params
			
			
	if args.max_atoms != 0:
		# We need to select a random subset of atoms to avoid using all of the
		# system memory.

		for kleft, kright in zip(left_group_data, right_group_data):
			l = left_group_data[kleft]
			if len(l) > args.max_atoms:
				choice_indices = np.random.choice(range(len(l)), args.max_atoms, replace=False)
				selection      = np.array(l)[choice_indices].tolist()
				left_group_data[kleft] = selection

			r = right_group_data[kright]
			if len(r) > args.max_atoms:
				choice_indices = np.random.choice(range(len(r)), args.max_atoms, replace=False)
				selection      = np.array(r)[choice_indices].tolist()
				right_group_data[kright] = selection


	# Now we need to build data structures that are applicable for two things.
	#     1) Taking the mean pearson coefficient for all unique combinations
	#        of gi's for two atoms with each dataset.
	#     2) Taking the mean pearson coefficient for all combinations of an
	#        atom in a left dataset and an atom in a right dataset.

	# The first list we need is all combinations of two unique atoms lined
	# up in two parallel arrays, one for each group. This will be used for
	# calculating self correlations.

	left_self_correlations  = []
	right_self_correlations = []

	for kleft, kright in zip(left_group_data, right_group_data):
		# Setup the left array.
		data = left_group_data[kleft]
		left_self_correlations.append(pearson(*construct_self_correlation_arrays(data, params_per_atom)))

		# Setup the left array.
		data = right_group_data[kright]
		right_self_correlations.append(pearson(*construct_self_correlation_arrays(data, params_per_atom)))

	# Now that the self correlation arrays are ready, we need to setup
	# the arrays for calculating the correlation between one group and 
	# another group.

	# We are going to put these arrays in the same order as the inputs
	# specified by the user.
	combo_arrays = []

	for kleft, kright in zip(args.lgroups, args.rgroups):
		left_data  = left_group_data[kleft]
		right_data = right_group_data[kright]

		size        = params_per_atom * len(left_data) * len(right_data)
		left_array  = np.zeros(size)
		right_array = np.zeros(size)

		idx = 0
		for l in range(len(left_data)):
			for r in range(len(right_data)):
				base   = idx * params_per_atom
				stride = base + params_per_atom

				left_array[base:stride]  = left_data[l]
				right_array[base:stride] = right_data[r]

				idx += 1

		combo_arrays.append([left_array, right_array])

	# We now have all of the necessary arrays. We need to calculate the
	# coefficients for them and put the coefficients into an output file.

	for idx, (kleft, kright) in enumerate(zip(args.lgroups, args.rgroups)):
		result = {}
		result['left_group']  = kleft
		result['right_group'] = kright

		cross_correlation      = pearson(*combo_arrays[idx])
		left_self_correlation  = left_self_correlations[idx]
		right_self_correlation = right_self_correlations[idx]

		result['cross']      = cross_correlation
		result['left_self']  = left_self_correlation
		result['right_self'] = right_self_correlation

		fname = '%s%s_vs_%s.json'%(args.odir, kleft, kright)

		# Yes, I know there is some redundant processing here. 
		# I just realized it though and its faster to just keep it than it
		# is to rewrite it.

		with open(fname, 'w') as file:
			print("Writing File %02i"%idx)
			file.write(json.dumps(result))


	print("Done")




	




















