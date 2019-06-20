from sys import path
path.append('subroutines')

import matplotlib.pyplot as plt
import numpy             as np
import Util
import sys
from   TrainingSet      import TrainingSetFile
from   sklearn.ensemble import RandomForestClassifier


def GetStructureParams(fname):
	Util.init('whatever')
	tset = TrainingSetFile(fname)
	values = []
	for k in tset.training_structures.keys():
		for atom in tset.training_structures[k]:
			values.append(atom.structure_params)

	v = np.array(values)
	return v.transpose()

def labelize(a):
	return (a * 10000).astype(int)

def delabelize(a):
	return np.array(a).astype(float) / 10000

def GetRandomForestPrediction(a, b, test):
	cl = RandomForestClassifier(n_jobs=1, n_estimators=25)
	a  = np.array([a]).transpose()
	t  = np.array([test]).transpose()
	b  = labelize(b)
	cl.fit(a, b)
	bp = cl.predict(t)
	return np.array(delabelize(bp))

def rmse(a, b):
	sum_sqr = np.square(a - b).sum()
	return np.sqrt(sum_sqr / len(a))

if __name__ == '__main__':
	file   = sys.argv[1]
	params = GetStructureParams(file)

	p1 = 14
	p2 = 10

	grid         = np.mgrid[0:60, 0:60].swapaxes(0, 2).swapaxes(0, 1)
	m            = grid.shape[0]
	r, c         = np.triu_indices(m, 1)
	combinations = grid[r, c]

	left  = combinations[:,0].astype(np.int32)
	right = combinations[:,1].astype(np.int32)

	np.flip(left)
	np.flip(right)

	all_rmse = []

	for l, r in zip(left, right):
		indices = np.array(range(len(params[0])))

		train = np.random.choice(indices, len(indices) // 2, replace=False)
		test  = np.random.choice(indices, len(indices) // 2, replace=False)

		train_left  = params[l][train]
		train_right = params[r][train]

		test_left  = params[l][test]
		test_right = params[r][test]

		prediction = GetRandomForestPrediction(train_left, train_right, test_left)

		avg_distance = (test_right - prediction).mean()
		prediction  += avg_distance

		all_rmse.append(rmse(test_right, prediction))

		perfect   = plt.scatter(test_left, test_right, s=12)
		predicted = plt.scatter(test_left, prediction, s=3)

		plt.legend([perfect, predicted], ["Original", "Random Forest"])
		plt.title("Random Forest Regression of Feature vs. Feature (RMSE = %2.3f)"%rmse(test_right, prediction))
		plt.show()

		compare   = plt.scatter(test_right, prediction, s=3)
		plt.title("Real vs. Predicted")
		plt.show()

	print(np.array(all_rmse).mean())
	print(np.array(all_rmse).min())
	print(np.array(all_rmse).max())