from Config import *
import numpy as np
import Util
from   Util import log, log_indent, log_unindent

class NeuralNetwork:
	def __init__(self, path):
		log("Loading Neural Network")
		log_indent()
		log("Path = %s"%path)

		self.path       = path
		self.initialize()

		self.load()


		log("Network Properties:")
		log_indent()
		log(str(self))
		log_unindent()
		log('')

		log_unindent()

	def generateNetwork(self):
		# TODO: Verify that this code is correct.
		n_values = self.layer_sizes[0]

		previous = self.layer_sizes[0]
		for n in self.layer_sizes[1:]:
			n_values += n * previous + n
			previous = n

		# This will automatically generate a list of values of length
		# n_values.
		self.network_values = np.random.uniform(-self.max_random, self.max_random, (n_values, 1))

		# Now that we have some values, load them into 
		# and appropriate structure.
		self.loadNetwork()


	# This function loads the actual weights and biases of
	# the neural network into a useable structure. It starts
	# by reading the list of raw values into a flat array and
	# then places them in an array that has 3 dimensions.
	# 
	# Array Format:
	#     1st index corresponds to the layer of the network
	#     2nd index corresponds to the node within that layer
	#     
	#      within each node, the following indices contain more information:
	#          0 = The weights of the node, in order, corresponding to the
	#              node in the previous layer that they connect from
	#          1 = The bias of the node.
	def loadNetwork(self):
		# This array stores the raw network values
		# in a flat array.
		self.network_values = []
		try:
			for idx, line in enumerate(self.lines[6:]):
				self.network_values.append(float(Util.GetLineCells(line)[0]))
		except ValueError as ex:
			raise ValueError("Unparseable value found on line %i of neural network file %s"%(7 + idx, self.path)) from ex

		# We need to ensure that an appropriate number of values exist in the
		# flat array we just loaded. 
		appropriate_value_count = 0
		for idx in range(len(self.layer_sizes)):
			if idx != 0:
				previous_layer_size = self.layer_sizes[idx - 1]
				current_layer_size  = self.layer_sizes[idx]
				appropriate_value_count += (previous_layer_size * current_layer_size) + current_layer_size

		if appropriate_value_count != len(self.network_values):
			e_str  = "The number of neural network values specified in the file does not match the network structure.\n"
			e_str += "%i values were present, but %i were expected."%(len(self.network_values), appropriate_value_count)
			raise ValueError(e_str)

		# Now that we have the flat array of values we can parse it according
		# to the format specified in the PDF James sent me.
		# TODO: Document this on Github.

		# This array will store the useful network structure.
		# It is fairly well formatted for conversion into a
		# PyTorch neural network. This line just makes it 
		# into an array containing one empty array per layer,
		# to be populated soon.
		self.layers = [[] for i in range(1, self.n_layers)]

		# The weights are stored in the network file with the following order:
		#     All of the weights coming from input one are stored first,
		#     in order by which node in the next layer they connect to.
		#     After all of these weights, the biases of the first actual
		#     layer are stored.

		weight_start_offset = 0

		# We need to iterate over each layer and load the corresponding
		# weights for each node in the layer (as well as the biases).
		for layer_size, idx in zip(self.layer_sizes, range(self.n_layers)):
			# The first "layer" is actually just the inputs, which don't have
			# biases, because this isn't really a layer.

			if idx != 0:
				previous_layer_size = self.layer_sizes[idx - 1]
				# We don't carry out this process on the first layer
				# for the above mentioned reason.

				# Here we create an array for each node in this layer.
				# This first index of the array will contain the weights,
				# the second will contain the bias.
				for i in range(layer_size):
					# Skip the first layer, as it is not a real layer.
					self.layers[idx - 1].append([[], 0.0])

				# Weight start offset should have been incremented by the previous
				# iteration of the loop, and we should be able to start reading
				# from whatever value it holds.

				# Each weight connected to this node is stored at an offset 
				# equal to the index of the node within the layer.
				# For example, if this is the first node in this layer,
				# the weight from the first node in the previous layer, to
				# this node will be at self.network_values[weight_start_offset + 0]
				# The weight from the second node in the first layer, to this node will
				# be at self.network_values[weight_start_offset + 16 + 0] if this layer
				# has 16 nodes.
				# If this is the second node in the layer, the above values would be
				# self.network_values[weight_start_offset + 1] and
				# self.network_values[weight_start_offset + 16 + 1] respectively.

				# I'm considering moving the following loops into their own function,
				# but I'm hesitant because there are so many state variables that would
				# need to be passed to the function and it will look really cluttered.

				for node_index in range(layer_size):
					# For each node in this layer, retrieve the
					# set of weights.
					for weight_index in range(previous_layer_size):
						# self.layers[idx - 1][node_index][0] is the
						# list of weights corresponding to the current node
						# in the current layer.
						offset               = weight_start_offset + weight_index*layer_size + node_index
						#print("offset: %i, start: %i, weight_idx: %i, lyr_size: %i, node_idx: %i"%(offset, weight_start_offset, weight_index, previous_layer_size, node_index))
						current_weight_value = self.network_values[offset]
						self.layers[idx - 1][node_index][0].append(current_weight_value)

				# Now that the weights for each node in the layer are read, we
				# need to load the biases.
				bias_offset = previous_layer_size * layer_size
				for node_index in range(layer_size):
					# For each node in this layer, retrieve the bias.
					offset = weight_start_offset + bias_offset + node_index
					self.layers[idx - 1][node_index][1] = self.network_values[node_index]

				# We have now loaded all of the relevant information for the layer.
				# We need to update weight_start_offset so that the next layer
				# starts from the appropriate offset.
				weight_start_offset = weight_start_offset + bias_offset + layer_size

	def load(self):
		# TODO: Document this format on Github.
		# This function actually parses all of the values in the header
		# of the neural network file. 

		# Read the configuration values in line 1.

		line1 = Util.GetLineCells(self.lines[0])

		self.gi_mode = int(line1[0])

		if self.gi_mode not in [0, 1]:
			raise ValueError("Invalid value specified for Gi mode on line 1 of %s"%(self.path))

		self.gi_shift = float(line1[1])
		if self.gi_shift not in [0.0, 0.5]:
			raise ValueError("Invalid value specified for reference Gi (shift) on line 1 of %s"%(self.path))

		self.activation_function = int(line1[2])
		if self.activation_function not in [0, 1]:
			raise ValueError("Invalid value specified for activation function on line 1 of %s"%(self.path))


		# Lines 2 and 3 just get copied into an output file, they
		# do not need to be parsed.

		self.line2 = self.lines[1]
		self.line3 = self.lines[2]

		# Read the configuration values in line 4

		line4 = Util.GetLineCells(self.lines[3])

		self.randomize = int(line4[0])
		if self.randomize not in [0, 1]:
			raise ValueError("Invalid value specified for randomization on line 4 of %s"%(self.path))

		self.randomize = self.randomize == 1

		try:
			self.max_random          = float(line4[1])
			self.cutoff_distance     = float(line4[2])
			self.truncation_distance = float(line4[3])
			self.gi_sigma            = float(line4[4])
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 4 of %s"%(self.path)) from ex

		line5 = Util.GetLineCells(self.lines[4])

		try:
			self.n_r0 = int(line5[0])

			self.r0   = []
			for i in line5[1:]:
				self.r0.append(float(i))
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 5 of %s"%(self.path)) from ex

		if len(self.r0) != self.n_r0:
			raise ValueError("The number of r0 values declared does not match the actual number present in the file. (Line 5 of %s)"%self.path)


		line6 = Util.GetLineCells(self.lines[5])

		try:
			self.n_layers = int(line6[0])

			self.layer_sizes = []
			for i in line6[1:]:
				self.layer_sizes.append(int(i))
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 6 of %s"%(self.path)) from ex

		if POTENTIAL_TYPE == 2 and self.layer_sizes[-1] != 8:
			raise ValueError("The specified potential type was BOP but the output layer of the neural network did not have 8 nodes.")
		elif POTENTIAL_TYPE == 1 and self.layer_sizes[-1] != 1:
			raise ValueError("The specified potential type was straight neural network but the output layer of the neural network ddi not have 1 node.")

		if self.layer_sizes[0] != self.n_r0 * 5:
			raise ValueError("The input layer dimensions of the neural network do not match the structural parameter dimensions.")

		if len(self.r0) != self.n_r0:
			raise ValueError("The number of layers declared does not match the actual number present in the file. (Line 6 of %s)"%self.path)

		# Here we actually either read or generate the neural network, based
		# on previsouly parsed values.
		if self.randomize:
			self.generateNetwork()
		else:
			self.loadNetwork()

	def initialize(self):
		# This function just performs basic file loading tasks.

		try:
			file = open(self.path, 'r')
		except FileNotFoundError as fer:
			raise Exception("The specified neural network file (%s) was not found."%self.path) from fer
		except Exception as ex:
			raise Exception("An error occured attempting to open the neural network file (%s)."%self.path) from ex

		try:
			raw_text = file.read()
		except Exception as ex:
			raise Exception("The training set file was opened but an error occured while reading it. File path (%s)."%self.path) from ex

		# Make sure line endings are correct.
		if '\r\n' in raw_text:
			if NORMALIZE_LINE_ENDINGS:
				print("This file contains non-unix line endings. They are being normalized automatically.")
				print("Set NORMALIZE_LINE_ENDINGS = False in %s "%CONFIG_FNAME)
				print("to disable this warning and fail when non-unix line endings are found.")
				raw_text = Util.NormalizeLineEndings(raw_text)
			else:
				raise Exception("This file contains non-unix line endings. Please convert this files line endings before use.")

		raw_text = raw_text.rstrip()
		lines = raw_text.split('\n')

		if WARN_ON_WHITESPACE_IN_TRAINING_SET:
			for i in lines:
				if i.isspace() or i == '':
					print("WARNING: This neural network file appears to contain unecessary whitespace.")
					print("Set WARN_ON_WHITESPACE_IN_TRAINING_SET = False in %s "%CONFIG_FNAME)
					print("to disable this warning.")

		self.lines = lines

	def __str__(self):
		# This is a built-in function that should be defined for all classes. Python automatically
		# calls this to convert an object to a string whenever an object is passed to a function
		# that normally takes a string as an argument. An example would be print(object)
		result  = "Gi Mode             = %s\n"%('Normal' if self.gi_mode == 0 else 'Shifted')
		result += "Gi Shift            = %f\n"%self.gi_shift
		result += "Activation Function = %s\n"%('Sigmoid' if self.activation_function == 0 else 'Shifted Sigmoid')
		result += "Network Randomized  = %s\n"%('Yes' if self.randomize else 'No (Read from file)')
		result += "Max Random Init.    = %f\n"%self.max_random
		result += "Cutoff Radius       = %f\n"%self.cutoff_distance
		result += "Truncation Distance = %f\n"%self.truncation_distance
		result += "Gi Sigma            = %f\n"%self.gi_sigma
		result += "R0 Values           = %s\n"%(' '.join([str(i) for i in self.r0]))
		result += "Layer Dimensions    = %s\n"%(' '.join([str(i) for i in self.layer_sizes]))
		return result


