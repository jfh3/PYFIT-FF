from   Config        import *
from   TrainingSet   import TrainingSetFile
from   NeuralNetwork import NeuralNetwork
import Util
import numpy as np
from   Util     import log, log_indent,log_unindent, ProgressBar
from   datetime import datetime

def generateNeighborList(poscar_data, nn):
	starttime = datetime.now()
	log("Generating Neighbor List")
	log_indent()
	log("Neighbor List Generation Started   at %s"%(starttime.strftime("%Y-%m-%d %H:%M:%S")))

	structures = poscar_data.structures
	
	# For each atom within each structure, we need to generate a list
	# of atoms within the cutoff distance. Periodic images need to be
	# accounted for during this process. Neighbors in this list are
	# specified as coordinates, rather than indices.

	# The final return value of this function in a 3 dimensional list,
	# with the following access structure: neighbor = list[structure][atom][neighbor_index]

	# First we will compute the total number of atoms that need to be
	# processed in order to get an estimate of the time this will take
	# to complete.
	n_total = sum([struct.n_atoms for struct in structures])

	bar = ProgressBar("Neighbor List ", 30, n_total, update_every = 25)

	# 1.5 for some things
	cutoff = nn.config.cutoff_distance * 1.0

	n_processed = 0

	neigborLists = []
	for structure in structures:
		# Normalize the translation vectors.
		A1_n = np.linalg.norm(structure.A1)
		A2_n = np.linalg.norm(structure.A2)
		A3_n = np.linalg.norm(structure.A3)

		# Numpy will automatically convert these to arrays when they are passed to
		# numpy functions, but it will do that each time we call a function. Converting
		# them beforehand will save some time.
		A1 = np.array(structure.A1)
		A2 = np.array(structure.A2)
		A3 = np.array(structure.A3)

		# Determine the number of times to repeat the
		# crystal structure in each direction.

		x_repeat = int(np.ceil(cutoff / A1_n))
		y_repeat = int(np.ceil(cutoff / A2_n))
		z_repeat = int(np.ceil(cutoff / A3_n))

		# Now we construct an array of atoms that contains all
		# of the repeated atoms that are necessary. We need to 
		# repeat the crystal structure from -repeat*A_n to 
		# positive repeat*A_n. 

		# This is the full periodic structure that we generate.
		# It is a list of vectors, each vector being a length 3
		# list of floating points.
		periodic_structure = []
		for i in range(-x_repeat, x_repeat + 1):
			for j in range(-y_repeat, y_repeat + 1):
				for k in range(-z_repeat, z_repeat + 1):
					# This is the new location to use as the center
					# of the crystal lattice.
					center_location = A1*i + A2*j + A3*k

					# Now we add each atom + new center location
					# into the periodic structure.
					for atom in structure.atoms:
						periodic_structure.append(atom + center_location)


		struct = np.array(periodic_structure)
		structure_neighbor_list = []

		# Here we actually iterate over every atom and then for each atom
		# determine which atoms are within the cutoff distance.
		for atom in structure.atoms:
			# This statement will subtract the current atom position from
			# the position of each potential neighbor, element wise. It will
			# then calculate the magnitude of each of these vectors element wise.
			distances = np.linalg.norm(
					struct - atom, 
					axis = 1
			)

			# This is special numpy syntax for selecting all items in 
			# an array that meet a condition. The boolean operators in
			# the square brackets actually convert the 'distances' array
			# into two arrays of boolean values and then computes their
			# boolean 'and' operation element wise. It then selects all 
			# items in the array 'struct' that correspond to a value of
			# true in the array of boolean values.
			neighbors     = struct[(distances > 0.0001) & (distances < cutoff)]
			# neighbor_vecs = neighbors - np.tile(atom, (len(neighbors), 1))

			# This line just takes all of the neighbor vectors that we now
			# have (as absolute vectors) and changes them into vectors 
			# relative to the atom that we are currently finding neighbors
			# for.
			neighbor_vecs = neighbors - atom

			structure_neighbor_list.append(neighbor_vecs)

		neigborLists.append(structure_neighbor_list)

		# Update the performance information so we can report
		# progress to the user.
		n_processed += structure.n_atoms
		bar.update(n_processed)
		

	
	bar.update(n_total)
	bar.finish()
	
	endtime = datetime.now()
	log("Neighbor List Generation Completed at %s"%(endtime.strftime("%Y-%m-%d %H:%M:%S")))
	log("Seconds Elapsed = %i\n"%((endtime - starttime).seconds))
	log_unindent()

	return neigborLists