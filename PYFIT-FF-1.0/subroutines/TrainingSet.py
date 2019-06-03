from Config import *
import Util
from   Util import log, log_indent, log_unindent
from   ConfigurationParser    import TrainingFileConfig

# Contains all of the data for a training set file.
# Can be sued to load data from a file and to dump
# data back into a file.
class TrainingSetFile:
	# Initializes a training set file instance from a file path.
	def __init__(self, path):
		log("Loading Training Set File")
		log_indent()
		log("Path = %s"%path)

		self.path       = path
		self.structures = []
		self.initialize()

		log_unindent()


	def initialize(self):
		# This function performs basic file loading tasks and exports the job
		# of parsing actual structures to another class.

		try:
			file = open(self.path, 'r')
		except FileNotFoundError as fer:
			raise Exception("The specified training set file (%s) was not found."%self.path) from fer
		except Exception as ex:
			raise Exception("An error occured attempting to open the training set file (%s)."%self.path) from ex

		try:
			raw_text = file.read()
		except Exception as ex:
			raise Exception("The training set file was opened but an error occured while reading it. File path (%s)."%self.path) from ex

		raw_text = raw_text.rstrip()
		self.lines = raw_text.split('\n')

		# Load all of the configuration information from the file.
		# Most of it isn't actually use for training, but should be
		# checked against the neural network to make sure that the
		# training configuration is sane.

		self.config = TrainingFileConfig(self.lines[:8], self.path)

		# Lines 7 and 8 contain the only two global configuration
		# values that we need.
		# TODO: Write code that checks all of these properties against
		#       those specified in the neural network file.
		self.n_structures    = int(Util.GetLineCells(self.lines[9])[1])
		self.n_atoms         = int(Util.GetLineCells(self.lines[10])[1])
		self.training_structures = {}

		# Every line from 13 onwards should correspond to a single atom.
		# Line 12 doesn't contain useful information.
		idx            = 12
		current_struct = []
		current_id     = 0
		while idx < len(self.lines):
			atom = TrainingInput(self.lines[idx], self.lines[idx + 1])
			if atom.structure_id != current_id:
				self.training_structures[current_id] = current_struct
				current_struct = []
				current_id     = atom.structure_id

			current_struct.append(atom)
			idx += 2

		self.training_structures[current_id] = current_struct
			

		# Make sure that the number of atoms and the number of
		# structures matches what was specified in the header file.
		# TODO: Implement this.

class TrainingInput:
	def __init__(self, line1, line2):
		cells1 = Util.GetLineCells(line1)
		self.group_name        = cells1[1]
		self.group_id          = int(cells1[2])
		self.structure_id      = int(cells1[3])
		self.structure_n_atoms = int(cells1[4])
		self.structure_energy  = float(cells1[5])
		self.structure_volume  = float(cells1[6])
		self.structure_params  = []

		# All remaining values should be structure parameters.
		# TODO: Implement a check against r0 * # of Legendre
		#       polynomials.
		for i in Util.GetLineCells(line2)[1:]:
			self.structure_params.append(float(i))

	def __str__(self):
		# This is a built-in function that should be defined for all classes. Python automatically
		# calls this to convert an object to a string whenever an object is passed to a function
		# that normally takes a string as an argument. An example would be print(object)
		result  = ""
		result += "Group Name           = %s\n"%(self.group_name)
		result += "Group ID             = %s\n"%(self.group_id)
		result += "Structure ID         = %s\n"%(self.structure_id)
		result += "Structure Atom ID    = %s\n"%(self.structure_n_atoms)
		result += "DFT Energy           = %s\n"%(self.structure_energy)
		result += "Structure Volume     = %s\n"%(self.structure_volume)
		result += "Structure Parameters = %s\n"%(' '.join([str(i) for i in self.structure_params]))
		return result