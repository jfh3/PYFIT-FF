# Sample Usage: python3 graph_ev.py output/E_VS_V.txt.txt 2
# The last parameter is how many of the samples to plot, starting from the first.

import matplotlib.pyplot as plt
import numpy             as np
import sys

if __name__ == '__main__':
	fname = sys.argv[1]
	if len(sys.argv) > 2:
		max_plot = int(sys.argv[2])
	else:
		max_plot = 100000

	file  = open(fname, 'r')
	raw   = file.read()
	file.close()

	lines = [l.split(' ') for l in raw.split('\n') if not l.isspace() and l != '']
	width = len(lines[0])

	volumes  = []
	energies = []

	for line in lines:
		volumes.append([float(i) for i in line[:width // 2]])
		energies.append([float(i) for i in line[width // 2:]])


	n_plots = min([max_plot, len(volumes)])
	plots = []
	for i in range(n_plots):
		plots.append(plt.scatter(volumes[i], energies[i], s=6))

	legend = ["Sample " + str(i) for i in range(n_plots)]

	plt.legend(plots, legend)
	plt.xlabel("Volume")
	plt.ylabel("Energy")
	plt.title("Energy vs. Volume")
	plt.show()	