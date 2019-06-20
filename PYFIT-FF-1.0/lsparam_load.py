from sys import path
path.append('subroutines')

import Util
from   TrainingSet import TrainingSetFile


def get_structure_params(fname):
	tset = TrainingSetFile(fname)
	values = []
	for k in tset.training_structures.keys():
		for atom in tset.training_structures[k]:
			values.append(atom.structure_params)

	v = np.array(values)
	return v.transpose()