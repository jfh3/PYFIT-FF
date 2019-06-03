# Sample Usage: python3 graph_loss.py output/loss_log.txt



import matplotlib.pyplot as plt
import numpy             as np
import sys

if __name__ == '__main__':
	fname = sys.argv[1]
	file  = open(fname, 'r')
	raw   = file.read()
	file.close()

	error_values = [float(i) for i in raw.split('\n') if not i.isspace() and i != '']
	indices      = range(len(error_values))

	plt.scatter(indices, error_values, s=8)
	plt.xlabel("Training Iteration")
	plt.ylabel("Root Mean Squared Error")
	plt.title("Error vs. Iteration")
	plt.show()	