from Config import *
import Util
from   Util import log, log_indent,log_unindent, ProgressBar

# Contains all of the data for a training set file.
# Can be sued to load data from a file and to dump
# data back into a file.
class PoscarDataFile:
	# Initializes a training set file instance from a file path.
	def __init__(self, path):
		log("Loading Structure File")
		log_indent()
		log("Path = %s"%path)

		self.path       = path
		self.structures = []
		self.initialize()

		log("Structure File Contains %6i Structures"%len(self.structures))
		log("                        %6i Atoms"%self.n_atoms)

		log("First Entry in File:")
		log_indent()
		log(str(self.structures[0]))
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

		raw_text = raw_text.rstrip()
		lines = raw_text.split('\n')

		self.n_atoms = 0


		bar = ProgressBar("Poscar Data ", 30, len(lines), update_every = 50)
		# Now we isolate every structure and send it to the TrainingSetStructure
		# class for full parsing.
		start_line = 0
		while True:
			# We need to know the number of atoms in the file 
			# before we can send the proper string of text to
			# the parsing function.
			try:
				atoms_in_file = int(lines[start_line + 5])
			except Exception as ex:
				raise Exception("Unable to read the number of atoms in the structure on line %i."%(start_line + 6)) from ex

			try:
				# Select a chunk of lines corresponding to a structure and send it to 
				# the class that parses structures.
				structure_chunk = lines[start_line : start_line + 8 + atoms_in_file]
			except IndexError as ex:
				err = "The file appears to be truncated improperly. Attempting to read the POSCAR "
				err += "structure starting at line %i resulted in an index error."%(start_line + 1)
				raise Exception(err) from ex

			try:
				struct = TrainingSetStructure(structure_chunk)
				self.n_atoms += struct.n_atoms
				self.structures.append(struct)
			except ValueError as ex:
				raise Exception("Error occured in POSCAR structure starting on line %i."%(start_line + 1)) from ex


			start_line += 8 + atoms_in_file
			bar.update(start_line)

			if start_line >= len(lines):
				bar.finish()
				break



# Conatains the information from a single set within the file.
# More specfically, the lattice vectors, number of atoms, 
# whether to use scaled or cartesion coordinates and the
# location of each atom. 
class TrainingSetStructure:
	# Initializes the data from a list of strings passed to it. The list should
	# be the chunk of the file that is interpretable as a POSCAR formatted
	# structure with the energy of the structure on the last line. This list
	# should just be a list of strings corresponding to lines. No line endings 
	# should be included.
	# TODO: Ask James how this could possibly work with only one energy value.
	def __init__(self, lines):
		self.comment = lines[0]
		try:
			self.scale_factor  = float(lines[1])
			self.A1            = self.parseVector(lines[2], self.scale_factor)
			self.A2            = self.parseVector(lines[3], self.scale_factor)
			self.A3            = self.parseVector(lines[4], self.scale_factor)
			self.n_atoms       = int(lines[5])
		except ValueError as ex:
			raise Exception("There was an error parsing a value in a POSCAR structure.") from ex

		if lines[6][0] == 'c':
			self.is_cartesian = True
		elif lines[6][0] == 'd':
			self.is_cartesian = False
			print("WUT")
		else:
			raise ValueError("Invalid value encountered in POSCAR structure. Line 7 should start with 'c' or 'd'")


		self.atoms = []
		for i in lines[7:-1]:
			try:
				self.atoms.append(self.parseVector(i, self.scale_factor))
			except ValueError as ex:
				raise ValueError("Invalid value encountered for atomic coordinate in POSCAR structure.") from ex


		try:
			self.energy = float(lines[-1]) + (self.n_atoms * E_SHIFT)
		except ValueError as ex:
			raise ValueError("Invalid value encountered for structure energy in POSCAR structure.") from ex



	def parseVector(self, string, scale):
		# This function parses a vector supplied as a string of space separated floating
		# point values. It also scales the vector based on the supplied scale factor.

		cells = [s for s in string.split(' ') if s != '' and not s.isspace()]
		if len(cells) != 3:
			raise ValueError("Encountered a number of vector components other than 3. (N = %i)"%(len(cells)))

		return [float(cells[0])*scale, float(cells[1])*scale, float(cells[2])*scale]

	def __str__(self):
		# This is a built-in function that should be defined for all classes. Python automatically
		# calls this to convert an object to a string whenever an object is passed to a function
		# that normally takes a string as an argument. An example would be print(object)
		result =  "Comment      = %s\n"%self.comment
		result += "Scale Factor = %f\n"%self.scale_factor
		result += "A1           = [%3.2f, %3.2f, %3.2f]\n"%(self.A1[0], self.A1[1], self.A1[2])
		result += "A2           = [%3.2f, %3.2f, %3.2f]\n"%(self.A2[0], self.A2[1], self.A2[2])
		result += "A3           = [%3.2f, %3.2f, %3.2f]\n"%(self.A3[0], self.A3[1], self.A3[2])
		result += "N Atoms      = %i\n"%self.n_atoms
		result += "Cartesian    = %s\n"%("Yes" if self.is_cartesian else "No")
		result += "DFT Energy   = %f\n"%self.energy
		result += "Atomic Locations: \n"
		for atom in self.atoms:
			result += "\t[%3.2f, %3.2f, %3.2f]\n"%(atom[0], atom[1], atom[2])

		return result

		






