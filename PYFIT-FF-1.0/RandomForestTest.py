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
	cl = RandomForestClassifier(n_jobs=1, n_estimators=8)
	a  = np.array([a]).transpose()
	t  = np.array([test]).transpose()
	b  = labelize(b)
	cl.fit(a, b)
	bp = cl.predict(t)
	return np.array(delabelize(bp))

def rmse(a, b):
	sum_sqr = np.square(a - b).sum()
	return np.sqrt(sum_sqr / len(a))

def rmse_norm(a, b):
	return rmse(a, b) / b.std()

def pearson(a, b):
	l = a - a.mean()
	r = b - b.mean()
	return (l*r).mean() / (a.std() * b.std())

def cf(a, b):
	return 1.0 - min(rmse_norm(a, b), 1.0)

if __name__ == '__main__':
	file   = sys.argv[1]
	params = GetStructureParams(file)

	p1 = 0
	p2 = 1


	perfect   = plt.scatter(params[p1], params[p2], s=1)
	plt.show()

	grid         = np.mgrid[0:60, 0:60].swapaxes(0, 2).swapaxes(0, 1)
	m            = grid.shape[0]
	r, c         = np.triu_indices(m, 1)
	combinations = grid[r, c]

	left  = combinations[:,0].astype(np.int32)
	right = combinations[:,1].astype(np.int32)

	np.flip(left)
	np.flip(right)

	all_rmse = []

	# Test completely random data.
	a = np.random.normal(0.0, 1.0, 4000)[::2]
	b = np.random.normal(0.0, 2.0, 4000)[::2]
	p = np.random.normal(0.0, 2.0, 4000)[::2]
	prediction = GetRandomForestPrediction(a, b, p)

	perfect   = plt.scatter(a, b, s=12)
	predicted = plt.scatter(a, prediction, s=3)

	plt.legend([perfect, predicted], ["Original", "Random Forest"])
	plt.title("Random Forest Regression of Random Data (Score = %2.3f, P = %2.3f)"%(cf(b, prediction), pearson(b, prediction)))
	plt.show()

	a = np.linspace(-1.0, 1.0, 4000)
	b = np.random.normal(0.0, 2.0, 4000) + np.linspace(-10.0, 10.0, 4000)
	a_t = a[::2]
	b_t = b[::2]
	a_v = a[1::2]
	b_v = b[1::2]
	prediction = GetRandomForestPrediction(a_t, b_t, a_v)

	perfect   = plt.scatter(a_t, b_t, s=12)
	predicted = plt.scatter(a_v, prediction, s=3)

	plt.legend([perfect, predicted], ["Original", "Random Forest"])
	plt.title("Random Forest Regression of Semi - Random Data (Score = %2.3f, P = %2.3f)"%(cf(b_v, prediction), pearson(b_v, prediction)))
	plt.show()


	for l, r in zip(left, right):
		indices = np.array(range(len(params[0])))

		train_left  = params[l][::2*10]
		train_right = params[r][::2*10]

		test_left  = params[l][1::2*10]
		test_right = params[r][1::2*10]

		prediction = GetRandomForestPrediction(train_left, train_right, test_left)

		avg_distance = (test_right - prediction).mean()
		prediction  += avg_distance

		all_rmse.append(rmse_norm(test_right, prediction))

		perfect   = plt.scatter(test_left, test_right, s=12)
		predicted = plt.scatter(test_left, prediction, s=3)

		plt.legend([perfect, predicted], ["Original", "Random Forest"])
		plt.title("Random Forest Regression of Feature vs. Feature (score = %2.3f, P = %2.3f)"%(cf(test_right, prediction), pearson(test_right, prediction)))
		plt.show()

		compare   = plt.scatter(test_right, prediction, s=3)
		plt.title("Real vs. Predicted")
		plt.show()

	print(np.array(all_rmse).mean())
	print(np.array(all_rmse).min())
	print(np.array(all_rmse).max())