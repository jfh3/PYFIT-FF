from Config import *
import numpy as np
import Util
from   Util import log, log_indent, log_unindent
from   ConfigurationParser    import TrainingFileConfig

class NeuralNetwork:
	def __init__(self, path):
		log("Loading Neural Network")
		log_indent()
		log("Path = %s"%path)

		self.path       = path
		self.initialize()

		self.load()

		log("File Configuration: ")
		log_indent()
		log(str(self.config))
		log_unindent()

		log_unindent()

	def generateNetwork(self, just_randomize=False):
		if just_randomize:
			log("Randomizing Network")
		else:
			log("Generating Network")
		# TODO: Verify that this code is correct.
		n_values = 0

		previous = self.config.layer_sizes[0]
		for n in self.config.layer_sizes[1:]:
			n_values += n * previous + n
			previous = n

		# This will automatically generate a list of values of length
		# n_values.
		self.network_values = np.random.uniform(-self.config.max_random, self.config.max_random, (n_values, 1))
		self.network_values = [v[0] for v in self.network_values]
		

		# Now that we have some values, load them into 
		# and appropriate structure.
		self.loadNetwork()
		self.config.randomize = False

		if not just_randomize:
			self.writeNetwork(self.path)


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
		if not self.config.randomize:
			# We weren't supposed to randomize the network,
			# so load it from the file.
			self.network_values = []

			try:
				for idx, line in enumerate(self.lines[8:]):
					self.network_values.append(float(Util.GetLineCells(line)[0]))
			except ValueError as ex:
				raise ValueError("Unparseable value found on line %i of neural network file %s"%(7 + idx, self.path)) from ex

		# We need to ensure that an appropriate number of values exist in the
		# flat array we just loaded. 
		appropriate_value_count = 0
		for idx in range(len(self.config.layer_sizes)):
			if idx != 0:
				previous_layer_size = self.config.layer_sizes[idx - 1]
				current_layer_size  = self.config.layer_sizes[idx]
				appropriate_value_count += (previous_layer_size * current_layer_size) + current_layer_size

		if appropriate_value_count != len(self.network_values):
			e_str  = "The number of neural network values specified in the file does not match the network structure. "
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
		self.layers = [[] for i in range(1, self.config.n_layers)]

		# The weights are stored in the network file with the following order:
		#     All of the weights coming from input one are stored first,
		#     in order by which node in the next layer they connect to.
		#     After all of these weights, the biases of the first actual
		#     layer are stored.

		weight_start_offset = 0

		# We need to iterate over each layer and load the corresponding
		# weights for each node in the layer (as well as the biases).
		for layer_size, idx in zip(self.config.layer_sizes, range(self.config.n_layers)):
			# The first "layer" is actually just the inputs, which don't have
			# biases, because this isn't really a layer.

			if idx != 0:
				previous_layer_size = self.config.layer_sizes[idx - 1]
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
					self.layers[idx - 1][node_index][1] = self.network_values[offset]

				# We have now loaded all of the relevant information for the layer.
				# We need to update weight_start_offset so that the next layer
				# starts from the appropriate offset.
				weight_start_offset = weight_start_offset + bias_offset + layer_size

	# Loads global configuration information from the network and
	# calls the functions that load the network weights and biases.
	def load(self):
		# TODO: Document this format on Github.
		# This function actually parses all of the values in the header
		# of the neural network file. 

		self.config = TrainingFileConfig(self.lines[:8], self.path)

		# Here we actually either read or generate the neural network, based
		# on previsouly parsed values.
		if self.config.randomize:
			self.generateNetwork()
		else:
			self.loadNetwork()


	# Writes the network to a file, copying the first six lines
	# verbatim, as this program does not change them.
	# The weights and biases may change though.
	def writeNetwork(self, path):
		file = open(path, 'w')

		file.write(self.config.toFileString())

		# Now we write the weights by column, rather than
		# by row for each layer. The biases go at the end
		# for each layer.
		# len(layer[0][0]) is the width of the weight matrix  (N)
		# len(layer)    is the height of the weight matrix (M)
		for layer in self.layers:
			# Write the weights.
			for weight in range(len(layer[0][0])):
				for node in range(len(layer)):
					file.write(' %-+17.8E 0.0000\n'%(layer[node][0][weight]))

			# Write the biases.
			for node in range(len(layer)):
				file.write(' %-+17.8E 0.0000\n'%(layer[node][1]))

		file.close()

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
			file.close()
		except Exception as ex:
			raise Exception("The training set file was opened but an error occured while reading it. File path (%s)."%self.path) from ex

		

		raw_text = raw_text.rstrip()
		lines = raw_text.split('\n')

		self.lines = lines