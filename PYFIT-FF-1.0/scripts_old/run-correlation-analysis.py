import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import copy
import subprocess
from   time import time

from sys import path
path.append("../subroutines")

# This is used to load a list of available structures.
from TrainingSet import TrainingSetFile
import Util

def get_groups(fname):
	training_set = TrainingSetFile(fname)

	last_group          = training_set.training_structures[0][0].group_name
	ordered_group_names = [last_group]

	for struct_id in training_set.training_structures:
		current_structure = training_set.training_structures[struct_id]
		current_group     = current_structure[0].group_name
		if current_group != last_group:
			if current_group not in ordered_group_names:
				ordered_group_names.append(current_group)
			last_group = current_group

	return sorted(ordered_group_names)

def print_groups(groups):
	print("Group Names (%i):"%(len(groups)))
	for subslice in range((len(groups) // 3)):
		print('\n\t', end='')
		for group in groups[subslice::(len(groups) // 3)]:
			print('%-7s '%group, end='')

	print('')
		

if __name__ == '__main__':
	# Here we will essentially be running pyfit in a loop
	# with different command line parameters so that we can
	# vary the target error for different structural groups.
	
	# The command line arguments to this program should be
	# as follows, in order.
	#     0) Log file.
	#     1) The name of the folder to write output files into.
	#     2) A slice in the same format as a python slice, indicating
	#        which of the group names (sorted lexicographically) to
	#        run this process for.
	#     3) The training set file to use for loading group names.
	#     4) The default target for group error.
	#     5) The target to set for the group being analyzed.
	#     6) Any command line arguments to pass along to the pyfit-ff
	#        runs that will be executed. They will be passed after all
	#        regular pyfit arguments, but before --target-errors.

	log_file = sys.argv[1]
	out_dir  = sys.argv[2]

	if out_dir[-1] != '/':
		out_dir += '/'

	left, right = sys.argv[3].split(':')
	if left == '' and right == '':
		section = slice(0, None) # The whole list
	elif left == '':
		section = slice(0, int(right))
	elif right == '':
		section = slice(int(left), None)
	else:
		section = slice(int(left), int(right))

	training_set_file = sys.argv[4]
	default_target    = float(sys.argv[5])
	main_target       = float(sys.argv[6])
	if len(sys.argv) >= 8:
		other_args = sys.argv[7:]

	# Now that the arguments are parse, display them.
	print("The following arguments have been recognized: ")
	print('\tLog File         :: %s'%log_file)
	print('\tOutput Directory :: %s'%out_dir)
	print('\tSlice            :: %s'%str(section))
	print('\tTraining Set     :: %s'%training_set_file)
	print('\tDefault Target   :: %f'%default_target)
	print('\tPrimary Target   :: %f'%main_target)
	if len(sys.argv) >= 8:
		print('\tAdditional Args  :: %s'%' '.join(other_args))

	# Initialize Util so that the TrainingSetFile object
	# can log what it is doing.
	Util.init(log_file)
	Util.set_mode(unsupervised=True)


	# Load the training set and get a sorted list of all groups
	# that it contains.
	groups = get_groups(training_set_file)

	# Display them so that the STDOUT log can be read later if
	# the user needs to make sure that everything is as expected
	print_groups(groups)

	Util.log("Beginning Training")
	Util.log_indent()

	# Now that everything is in order, start running pyfit-ff for every
	# group specified, each with their own output directory. This script
	# will defer to Config.py for most of the configuration parameters.
	for group in groups[section]:
		out_dir_name       = out_dir + group + '/'

		full_command_line  = 'python3 pyfit-ff.py -t -d '
		full_command_line += out_dir_name
		full_command_line += ' ' + ' '.join(other_args) + ' '
		full_command_line += '--target-errors %f %s:%f'%(default_target, group, main_target)
		
		stdout_name = out_dir_name + 'stdout'

		Util.log("COMMAND: %s"%full_command_line)

		process = subprocess.Popen(full_command_line.split(' '), stdout=subprocess.PIPE)
		output, error = process.communicate()

		if error != '' and error != None:
			print("ERROR: %s"%error)

		f = open(stdout_name, 'w')
		f.write(output.decode('utf-8'))
		f.close()

	Util.log_unindent()
		