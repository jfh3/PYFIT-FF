from   Config        import *
from   TrainingSet   import TrainingSetFile
from   NeuralNetwork import NeuralNetwork
import Util
import matplotlib.pyplot as plt
import numpy as np

from   Util     import log, log_indent,log_unindent, ProgressBar
from   datetime import datetime

def GenerateStructuralParameters(poscar_data, nl, nn):
	starttime = datetime.now()
	log("Generating Structural Parameters")
	log_indent()
	log("Structural Parameter Generation Started   at %s"%(starttime.strftime("%Y-%m-%d %H:%M:%S")))

	# This function is broken up into several loops. In principal,
	# it could be done without any loops by carefully applying numpy
	# operations. This would be really hard code to read though. I
	# believe that the way it is currently implemented strikes a 
	# good balance between readabililty and speed. Its about 150
	# times faster than a pure Python implementation.

	structs = poscar_data.structures

	# Here we compute the number of operations that will need
	# to take place in order to calculate the structural 
	# parameters. This is somewhat of an estimate, but the
	# operation should scale roughly by a factor of n^2.
	n_total     = 0
	for struct in nl:
		for atom in struct:
			n_total += len(atom)**2
	n_processed = 0

	bar = ProgressBar("Structural Parameters ", 30, n_total, update_every = 8)

	structural_parameters = []
	# Here we iterate over every structure. And then
	# over every atom. We export the structural parameter
	# calculation for each individual atom to another function.
	for i, struct in enumerate(nl):
		processed = 0
		# Iterate over all structures.
		parameters_for_structure = []
		for atom in struct:
			processed += len(atom)**2
			# Iterate over each atom in the structure and compute the (usually 60)
			# paramters for it.
			parameters_for_structure.append(compute_parameters(
				np.array(atom), 
				nn.config.r0, 
				nn.config.gi_sigma, 
				nn.config.cutoff_distance, 
				nn.config.truncation_distance, 
				nn.config.gi_mode, 
				nn.config.gi_shift
			))

		n_processed += processed
		bar.update(n_processed)

		structural_parameters.append(parameters_for_structure)

	bar.finish()

	endtime = datetime.now()
	log("Structural Parameter Generation Completed at %s"%(endtime.strftime("%Y-%m-%d %H:%M:%S")))
	log("Seconds Elapsed = %i\n"%((endtime - starttime).seconds))
	log_unindent()


	return structural_parameters

# This funciton takes the neighbor list for a single atom 
# and computes the structural parameters for it.
def compute_parameters(atom, r0, sigma, cutoff, truncation, gi_mode, gi_shift):
	# First we need a list of every unique combination of
	# two neighbors, not considering [0, 1] to be unique 
	# compared to [1, 0]. More specifically, a different
	# order does not make the pair unique.

	# This list of combinations is stored as indices.
	# After it is built, we will use it to produce 
	# two parallel arrays, that will be well formed 
	# for use of numpy routines to speed up the structural
	# parameter calculations.
	combinations = []
	length = len(atom)


	# This is a little hard to interpret. Conceptualize
	# combinations of i and j atoms in the neighbor list as
	# a square matrix with side length equal to the number of
	# neighbors, and each element being a pair of indices, equal
	# to [m, n], with m and n being the column and row of the
	# matrix. If we select all combinations above a diagonal, NOT
	# including the diagonal itself, we get all unique combinations
	# of i and j, not including where i = j or where i and j are 
	# swapped.
	#
	# This set of numpy expressions will create this list and
	# it does it very quickly, because internally, all of these
	# functions are implemented in c.
	grid         = np.mgrid[0:length, 0:length].swapaxes(0, 2).swapaxes(0, 1)
	m            = grid.shape[0]
	r, c         = np.triu_indices(m, 1)
	combinations = grid[r, c]


	# Now we have numpy select two arrays of vectors. 
	# Left array is the vector corresponding to the first index
	# in each pair.
	# Right array is the vector corresponding to the second index
	# in each pair.
	left_array  = atom[combinations[:,0]]
	right_array = atom[combinations[:,1]]

	# Now we use these pairs of vectors to compute and array of
	# cos(theta) values.
	dot_products       = np.einsum('ij,ij->i', left_array, right_array)


	# This is the magnitude of all of the vectors in the left array.
	left_magnitudes    = np.linalg.norm(left_array, axis=1)

	# This is the magnitude of all of the vectors in the right array.
	right_magnitudes   = np.linalg.norm(right_array, axis=1)

	# The following two lines are essentially computing (r_i * r_j) / (|r_i||r_j|)
	# where '*' denotes the dot product.
	magnitude_products = left_magnitudes * right_magnitudes
	angular_values     = dot_products / magnitude_products

	# Here we skip some steps and just add an array of 1.0 onto
	# the array of cos(theta) values. This is for all cases where
	# i = j, so we know for a fact that theta = 0 and cos(theta) = 1.0
	dupl_indices    = np.arange(0, length, 1)
	dupl_magnitudes = np.linalg.norm(atom[dupl_indices], axis=1)
	angular_values  = np.concatenate((angular_values, np.tile([1.0], length)))

	# angular values now holds an array of cos(theta_ijk) for all unique i, j.

	# Next, we need to compute and array of radial terms for each r0 value.
	s2 = 1.0/(sigma**2)
	s3 = 1.0/(sigma**3)

	# This is an array of all radial terms for 
	# all values of r0.
	radial_terms = []

	# All of the following mathematical operations are part of the 
	# structural parameter calculation method defined in the files
	# that were sent to me. These operations are not done inside of
	# the subsequent loop, because their values do not vary with
	# respect to r0. It is worth noting that you could do this inside
	# of the loop without any slowdown, but that is because numpy
	# will cache the values and does not compute them again when
	# it doesn't need to.

	# The computation involving tanh at the end of the cutoff function
	# terms is just a mathematical way of making fc be zero if r > rc.
	# Adding an if statement would require numpy to jump out of c code and
	# in to python code in order to evaluate it. This would significantly
	# slow down the operation. (During testing slowdown was 50 - 100 times)
	# see https://www.desmos.com/calculator/puz9hpi090
	d4 = np.square(np.square(truncation))
	left_r_rc_unmodified = left_magnitudes - cutoff
	left_r_rc_terms = np.square(np.square(left_r_rc_unmodified))
	left_fc         = (left_r_rc_terms / (d4 + left_r_rc_terms))*(0.5*np.tanh(-1e6*(left_r_rc_unmodified)) + 0.5)

	right_r_rc_unmodified = right_magnitudes - cutoff
	right_r_rc_terms = np.square(np.square(right_r_rc_unmodified))
	right_fc         = (right_r_rc_terms / (d4 + right_r_rc_terms))*(0.5*np.tanh(-1e6*(right_r_rc_unmodified)) + 0.5)

	r_rc_unmodified = dupl_magnitudes - cutoff
	r_rc_terms      = np.square(np.square(r_rc_unmodified))
	fc              = (r_rc_terms / (d4 + r_rc_terms))*(0.5*np.tanh(-1e6*(r_rc_unmodified)) + 0.5)

	


	# Here we calculate the radial term for all values of r0.
	for r0n in r0:
		
		# The left_* and right_* arrays correspond to cases where 
		# r_i != r_j. In these cases, we need to calculate both of 
		# the functions (f) independently.
		left_term       = s3*np.exp(-s2*np.square(left_magnitudes - r0n))
		full_left_term  = left_term*left_fc

		right_term      = s3*np.exp(-s2*np.square(right_magnitudes - r0n))
		full_right_term = right_term*right_fc

		# These two arrays correspond to cases where r_i = r_j and we 
		# know that we just need to square the value of the function 
		# (f) after computing it once.
		term            = s3*np.exp(-s2*np.square(dupl_magnitudes - r0n))
		full_term       = term*fc

		# In this statement, we multiply the radial term by 2, because 
		# cases where r_i != r_j are supposed to be repeated, with the 
		# vectors swapped. Since the function we are computing on them
		# is symmetric with respect to the order of its arguments, we 
		# can just compute one case of r_i != r_j and double it to 
		# account for the case where r_i is swapped with r_j. This cuts
		# the computation time almost in half.
		to_add = np.concatenate((2 * full_right_term * full_left_term, np.square(full_term)))
		radial_terms.append(to_add)

		


	# Now radial_terms is an array where each first index corresponds 
	# to an r0 value and each second index corresponds to the product
	# of the radial terms for a unique combination of neighbors.

	# For each r0 and for each combination of neigbors, we now
	# Need to compute the m-th Legendre polynomial of the cosine
	# of the angle between the two.

	# The computations within these loops are all optimized variations
	# of standard Legendre polynomials. The most important optimization
	# is that the squares are pre-computed outside of the loops to
	# reduce unecessary computations.

	legendre_0_values = []
	for r0n in radial_terms:
		legendre_0_values.append(1 * r0n)

	legendre_1_values = []
	for r0n in radial_terms:
		legendre_1_values.append(angular_values * r0n)

	legendre_2_values = []
	sq = np.square(angular_values)
	for r0n in radial_terms:
		legendre_2_values.append((1.5*sq - 0.5) * r0n)

	legendre_4_values = []
	for r0n in radial_terms:
		legendre_4_values.append((sq*(4.375*sq - 3.75) + 0.375) * r0n)

	legendre_6_values = []
	for r0n in radial_terms:
		legendre_6_values.append((sq*(sq*(14.4375*sq - 19.6875) + 6.5625) - 0.3125) * r0n)

	# Now we have an array of P_m(cos(theta_ijk))*f(r_ij)*f(r_ik) for every r0 value
	# and for every unique neighbor combination. Now we just need to sum over every 
	# neighbor comination and we are done.

	structural_parameters = []
	for r0n in legendre_0_values:
		structural_parameters.append(np.sum(r0n))

	for r0n in legendre_1_values:
		structural_parameters.append(np.sum(r0n))

	for r0n in legendre_2_values:
		structural_parameters.append(np.sum(r0n))

	# for r0n in legendre_3_values:
	# 	structural_parameters.append(np.sum(r0n))

	for r0n in legendre_4_values:
		structural_parameters.append(np.sum(r0n))

	for r0n in legendre_6_values:
		structural_parameters.append(np.sum(r0n))

	# TODO: Replace 5 with # of legendre polynomials
	

	if gi_mode == 1:
		sp = np.array(structural_parameters) / np.square(np.tile(r0, 5))
		return np.log(sp + gi_shift)
	else:
		sp = np.array(structural_parameters) / 16.0
		return sp