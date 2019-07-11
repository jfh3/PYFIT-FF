import matplotlib.pyplot as plt
import numpy             as np
import code
import argparse
import os
import time
import sys
import json

from sys import path
sys.path.append('scripts')

import tl

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Reads all of the output files from the given directory " +
		"and converts them into matrix files and label files suitable for use " +
		"with matrix_vis.py. This is meant to be run on the output of " +
		"structure-lsp-correlation-controller.py" 
	)

	parser.add_argument(
		'-i', '--in-dir', dest='input_directory', type=str, required=True,
		help='The directory that contains the data to analyze.'
	)

	parser.add_argument(
		'-o', '--out-dir', dest='output_directory', type=str, required=True,
		help='The directory to place the output files into.'
	)

	args = parser.parse_args()

	if not os.path.isdir(args.input_directory):
		print("Input directory inaccessible.")
		exit(1)

	if os.path.isdir(args.output_directory):
		print("Output directory already exists.")
		exit(1)

	os.mkdir(args.output_directory)


	if args.output_directory[-1] != '/':
		args.output_directory += '/'

	# Load all of the json files.

	_, files, _ = tl.walk_dir_recursive(args.input_directory, 2, extension='.json')

	all_file_contents = files[0]

	raw_data = [json.loads(f) for f in all_file_contents]

	# Now we need a list of unique names to base the rest
	# of this process off of.

	all_names = []

	for d in raw_data:
		lname = d['left_group']
		rname = d['right_group']

		if lname not in all_names:
			all_names.append(lname)

		if rname not in all_names:
			all_names.append(rname)

	all_names = sorted(all_names)

	# Now we have a unique, sorted list of names.
	# We need to setup a dictionary in which the first key is
	# the left group name and the second key is the right group
	# name. We will use this to construct a matrix for plotting.

	self_correlations = {}
	data_dict         = {}
	for name in all_names:
		# Find all items for which this is the left name.
		left_items = {}
		for item in raw_data:
			if item['left_group'] == name:
				self_correlations[name]         = item['left_self']
				left_items[item['right_group']] = item['cross']

			if item['right_group'] == name:
				self_correlations[name]         = item['right_self']
		data_dict[name] = left_items

	# We should now have len(all_names) entries in the dictionary.
	if len(data_dict) != len(all_names):
		print("SOMETHING IS WRONG")
		print(len(data_dict))
		print(len(all_names))
		exit(1)

	cross_matrix = np.ones((len(all_names), len(all_names)))

	for i, name in enumerate(all_names):
		for j, name2 in enumerate(all_names):
			value = 0.0
			if j == i:
				value = 1.0
			elif name in data_dict:
				value = data_dict[name]
				if name2 in value:
					value = value[name2]
				else:
					value = data_dict[name2][name]
			elif name2 in data_dict:
				value = data_dict[name2][name]
			else:
				print("SOMETHING ELSE IS WRONG")
				exit(1)
			cross_matrix[i, j] = value

	# We now have the raw values for the cross correlations.
	# Make a duplicate matrix where each value is based on the
	# cross correlation over the self correlation, whichever of
	# the two possible values is higher.

	relative_matrix = np.copy(cross_matrix)

	for i in range(relative_matrix.shape[0]):
		for j in range(relative_matrix.shape[1]):
			if i != j:
				# Find the self correlations for the matching indices.
				self01 = self_correlations[all_names[i]]
				self02 = self_correlations[all_names[j]]

				ratio01 = relative_matrix[i][j] / self01
				ratio02 = relative_matrix[i][j] / self02

				ratio = max([ratio01, ratio02])

				relative_matrix[i][j] = ratio

	# Export the files.
	tl.ndarray_to_file(cross_matrix, args.output_directory + 'cross.mat')
	tl.ndarray_to_file(relative_matrix, args.output_directory + 'relative.mat')

























