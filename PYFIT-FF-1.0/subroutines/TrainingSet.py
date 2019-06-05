from Config import *
import Util
import numpy as np
from   Util import log, log_indent, log_unindent, ProgressBar
from   ConfigurationParser    import TrainingFileConfig
from   datetime import datetime

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
		self.potential_type  = int(Util.GetLineCells(self.lines[8])[1])
		self.n_structures    = int(Util.GetLineCells(self.lines[9])[1])
		self.n_atoms         = int(Util.GetLineCells(self.lines[10])[1])
		self.training_structures = {}

		bar = ProgressBar("Loading Training Set", 30, self.n_structures, update_every = 3)

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
				bar.update(current_id + 1)

			current_struct.append(atom)
			idx += 2

		bar.finish()
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

# This function is designed to write the generated structure parameter
# information to a file in the format also used by this program to store
# and load training data. If the neighbor_lists parameter is specified,
# this function will write neighbor list data as well.
def WriteTrainingSet(fname, config, poscar_data, structure_params, neighbor_lists = None):
	starttime = datetime.now()
	if neighbor_lists == None:
		log("Writing LSParam File")
	else:
		log("Writing Neighbor List File")

	log_indent()
	log("Write Started   at %s"%(starttime.strftime("%Y-%m-%d %H:%M:%S")))


	file = open(fname, 'w')

	file.write(config.toFileString(prepend_comment = True))
	file.write(' # %i - Potential Type\n'%(1))
	file.write(' # %i - Number of Structures\n'%(len(poscar_data.structures)))
	file.write(' # %i - Number of Atoms\n'%(poscar_data.n_atoms))
	file.write(' # ATOM-ID GROUP-NAME GROUP_ID STRUCTURE_ID STRUCTURE_Natom STRUCTURE_E_DFT STRUCTURE_Vol\n')

	total_atom_idx = 0
	atom_idx       = 0
	structure_idx  = 0
	group_idx      = 1
	current_group  = poscar_data.structures[0].comment

	bar_title = 'Writing LSParams ' if neighbor_lists == None else 'Writing Neighbor Lists'
	bar = ProgressBar(bar_title, 30, poscar_data.n_atoms, update_every = 50)


	for struct in poscar_data.structures:
		if struct.comment != current_group:
			current_group = struct.comment
			group_idx += 1

		# This compute the parallelepiped volume of the 
		# structure based purely on its basis vectors.
		struct_volume = np.linalg.norm(
			np.cross(
				np.cross(struct.A1, struct.A2),
				struct.A3
			)
		)

		atom_idx = 0
		for atom in struct.atoms:
			file.write('ATOM-%i %s %i %i %i %.6E %.6E\n'%(
				total_atom_idx,
				current_group,
				group_idx,
				structure_idx,
				struct.n_atoms,
				struct.energy,
				struct_volume
			))

			params_string = ' '.join(['%.6E'%g for g in structure_params[structure_idx][atom_idx]])
			file.write('Gi  %s\n'%(params_string))

			if neighbor_lists != None:
				current_list = neighbor_lists[structure_idx][atom_idx]
				nbl_string   = ' '.join(['%.6E %.6E %.6E'%(n[0], n[1], n[2]) for n in current_list])
				file.write('NBL %i %s\n'%(len(current_list), nbl_string))

			total_atom_idx += 1
			atom_idx       += 1
		
		bar.update(total_atom_idx)

		structure_idx += 1

	bar.finish()
	file.write('\n')
	file.close()

	endtime = datetime.now()
	log("Write Completed at %s"%(endtime.strftime("%Y-%m-%d %H:%M:%S")))
	log("Seconds Elapsed = %i\n"%((endtime - starttime).seconds))
	log_unindent()