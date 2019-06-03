import Util

# This is used to parse the header of the neural network
# and LSParam files. 
class TrainingFileConfig:

	def __init__(self, lines):
		# First, convert the lines into arrays
		# of space separated values and remove any
		# preceeding '#' characters

		def remove_comment(line):
			return [c for c in Util.GetLineCells(line) if c != '#']

		line1 = remove_comment(lines[0])
		line2 = remove_comment(lines[1])
		line3 = remove_comment(lines[2])
		line4 = remove_comment(lines[3])
		line5 = remove_comment(lines[4])
		line6 = remove_comment(lines[5])
		line7 = remove_comment(lines[6])
		line8 = remove_comment(lines[7])

		self.gi_mode = int(line1[0])

		if self.gi_mode not in [0, 1]:
			raise ValueError("Invalid value specified for Gi mode on line 1 of %s"%(self.path))

		self.gi_shift = float(line1[1])
		if self.gi_shift not in [0.0, 0.5]:
			raise ValueError("Invalid value specified for reference Gi (shift) on line 1 of %s"%(self.path))

		self.activation_function = int(line1[2])
		if self.activation_function not in [0, 1]:
			raise ValueError("Invalid value specified for activation function on line 1 of %s"%(self.path))

		try:
			self.n_species = int(line2[0])
			self.element   = line3[0]
			self.mass      = float(line3[1])
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 2 or 3 of %s"%(self.path)) from ex

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

		# As per the new spec, line 5 contains the orders of legendre polynomials in use.
		try:
			self.n_legendre_polynomials = int(line5[0])
			self.legendre_orders        = [int(c) for c in line5[1:]]
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 5 of %s"%(self.path)) from ex

		if len(self.legendre_orders) != self.n_legendre_polynomials:
			error  = "Number of specified legendre polynomials does not match expected value."
			error += "%i were supposed to be given, but %i were specified."%(self.n_legendre_polynomials, len(self.legendre_orders)) 
			raise ValueError(error)

		try:
			self.n_r0 = int(line6[0])
			self.r0   = [float(c) for c in line6[1:]]
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 6 of %s"%(self.path)) from ex

		if len(self.r0) != self.n_r0:
			error  = "The number of r0 values declared does not match the actual number present in the file."
			error += "(Line 6 of %s)"%self.path
			raise ValueError(error)

		# Load the BOP parameters from line 7, even though we don't know what to 
		# do with them. They should be compared between the two files to make
		# sure that everything makes sense.

		try:
			self.BOP_param0     = int(line7[0])
			self.BOP_parameters = [float(c) for c in line7[1:]]
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 7 of %s"%(self.path)) from ex

		# Parse the network structure.
		try:
			self.n_layers    = int(line8[0])
			self.layer_sizes = [int(c) for c in line8[1:]]
		except ValueError as ex:
			raise ValueError("Unable to parse value on line 8 of %s"%(self.path)) from ex

		# Make sure that the network structure actaully matches the number of 
		# structure parameters that will be supplied to it.

		if self.layer_sizes[0] != self.n_r0 * self.n_legendre_polynomials:
			raise ValueError("The input layer dimensions of the neural network do not match the structural parameter dimensions.")

	def __str__(self):
		# This is a built-in function that should be defined for all classes. Python automatically
		# calls this to convert an object to a string whenever an object is passed to a function
		# that normally takes a string as an argument. An example would be print(object)
		result  = "Gi Mode             = %s\n"%('Normal' if self.gi_mode == 0 else 'Shifted')
		result += "Gi Shift            = %f\n"%self.gi_shift
		result += "Activation Function = %s\n"%('Sigmoid' if self.activation_function == 0 else 'Shifted Sigmoid')
		result += "# of Species        = %i\n"%self.n_species
		result += "Element             = %s\n"%self.element
		result += "Mass                = %s\n"%self.mass
		result += "Network Randomized  = %s\n"%('Yes' if self.randomize else 'No (Read from file)')
		result += "Max Random Init.    = %f\n"%self.max_random
		result += "Cutoff Radius       = %f\n"%self.cutoff_distance
		result += "Truncation Distance = %f\n"%self.truncation_distance
		result += "Gi Sigma            = %f\n"%self.gi_sigma
		result += "# Legendre Poly.    = %i\n"%self.n_legendre_polynomials
		result += "Legendre Orders     = %s\n"%(' '.join([str(i) for i in self.legendre_orders]))
		result += "R0 Values           = %s\n"%(' '.join([str(i) for i in self.r0]))
		result += "BOP Parameter 0     = %i\n"%self.BOP_param0
		result += "BOP Paramters       = %s\n"%(' '.join([str(i) for i in self.BOP_parameters]))
		result += "Layer Dimensions    = %s\n"%(' '.join([str(i) for i in self.layer_sizes]))
		return result


	def __eq__(self, other):
		# Don't compare the randomize flag.
		equality  = self.gi_mode                == other.gi_mode
		equality &= self.gi_shift               == other.gi_shift
		equality &= self.activation_function    == other.activation_function
		equality &= self.n_species              == other.n_species
		equality &= self.element                == other.element
		equality &= self.mass                   == other.mass
		equality &= self.max_random             == other.max_random
		equality &= self.cutoff_distance        == other.cutoff_distance
		equality &= self.truncation_distance    == other.truncation_distance
		equality &= self.gi_sigma               == other.gi_sigma
		equality &= self.n_legendre_polynomials == other.n_legendre_polynomials
		equality &= self.legendre_orders        == other.legendre_orders
		equality &= self.r0                     == other.r0
		equality &= self.BOP_param0             == other.BOP_param0
		equality &= self.BOP_parameters         == other.BOP_parameters
		equality &= self.layer_sizes            == other.layer_sizes

		return equality