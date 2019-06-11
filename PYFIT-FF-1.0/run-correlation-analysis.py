import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import copy
from   time import time

from sys import path
path.append("subroutines")

# This is used to load a list of available structures.
from TrainingSet import TrainingSetFile

if __name__ == '__main__':
	# Here we will essentially be running pyfit in a loop
	# with different command line parameters so that we can
	# vary the target error for different structural groups.
	
	# The command line arguments to this program should be
	# as follows, in order.
	#     1) The name of the folder to write output files into.
	#     2) A slice in the same format as a python slice, indicating
	#        which of the group names (sorted lexicographically) to
	#        run this process for.
	#     3) The training set file to use for loading group names.
	#     4) Any command line arguments to pass along to the pyfit-ff
	#        runs that will be executed. They will be passed after all
	#        regular pyfit arguments, but before --target-errors.

	out_dir  = sys.argv[1]

	left, right = sys.argv[2].split(':')
	section  = slice(int(left), int(right))

	training_set_file = sys.argv[3]
	if len(sys.argv) >= 5:
		other_args = sys.argv[4:]

	# Now that the arguments are parse, display them.
	print("The following arguments have been recognized: ")
	print('\tOutput Directory :: %s'%out_dir)
	print('\tSlice            :: %i:%i'%(section.start, section.stop))
	print('\tTraining Set     :: %s'%training_set_file)
	if len(sys.argv) >= 5:
		print('\tAdditional Args  :: ')
		for arg in other_args:
			print('\t\t%s'%arg)


	training_set = TrainingSetFile(training_set_file)

	last_group          = training_set.training_structures[0][0].group_name
	ordered_group_names = [last_group]

	for struct_id in training_set.training_structures:
		current_structure = training_set.training_structures[struct_id]
		current_group     = current_structure[0].group_name
		if current_group != last_group:
			if current_group not in ordered_group_names:
				ordered_group_names.append(current_group)
			last_group = current_group

	groups = sorted(ordered_group_names)

	print("Group Names:")
	for g in groups:
		print('\t%s'%g)