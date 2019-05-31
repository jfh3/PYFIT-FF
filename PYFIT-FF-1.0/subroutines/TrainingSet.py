from Config import *
import Util
from   Util import log, log_indent, log_unindent

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

		log("First Entry in File:")
		log_indent()
		log(str(self.training_inputs[0]))
		log_unindent()
		log('')

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
		self.lines = raw_text.split('\n')

		if WARN_ON_WHITESPACE_IN_TRAINING_SET:
			for i in self.lines:
				if i.isspace() or i == '':
					print("WARNING: This training set file appears to contain unecessary whitespace.")
					print("Set WARN_ON_WHITESPACE_IN_TRAINING_SET = False in %s "%CONFIG_FNAME)
					print("to disable this warning.")

		# The first six lines of the LSParam file are either redundant with the
		# neural network file or not relevant to this program.

		# Lines 7 and 8 contain the only two global configuration
		# values that we need.
		# TODO: Write code that checks all of these properties against
		#       those specified in the neural network file.
		self.n_structures    = int(Util.GetLineCells(self.lines[6])[1])
		self.n_atoms         = int(Util.GetLineCells(self.lines[7])[1])
		self.training_inputs = []

		# Every line from 10 onwards should correspond to a single atom.
		# Line 9 doesn't contain useful information.
		idx = 9
		while idx < len(self.lines):
			self.training_inputs.append(TrainingInput(self.lines[idx], self.lines[idx + 1]))
			idx += 2
			

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