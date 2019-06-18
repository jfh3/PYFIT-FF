import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import json
from   time          import time
from   datetime      import datetime
import warnings
warnings.filterwarnings("ignore")

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def GenHeatmap(results, output_file, abs_display):
	n_params_per_atom       = results["n_params_per_atom"]
	correlation_matrix      = np.zeros((n_params_per_atom, n_params_per_atom))

	axis_labels = []
	for l in results["legendre_polynomials"]:
		for r in results["r_0_values"]:
			label = "$P_" + str(l) + r"\;\;r = " + '%1.2f'%r + "$"
			axis_labels.append(label)

	
	# Construct an easy to graph matrix.
	for coeff in results["coefficients"]:
		l_idx = coeff["param0"]["idx"]
		r_idx = coeff["param1"]["idx"]
		pcc   = coeff["pcc"]

		if np.abs(pcc) > 1.0:
			eprint("ERROR: One of the coefficients was outside of [-1, 1]")
			sys.exit(1)

		if abs_display:
			correlation_matrix[l_idx][r_idx] = np.abs(pcc)
			correlation_matrix[r_idx][l_idx] = np.abs(pcc)
		else:
			correlation_matrix[l_idx][r_idx] = pcc
			correlation_matrix[r_idx][l_idx] = pcc


	for mn in range(n_params_per_atom):
		correlation_matrix[mn][mn] = 1.0

	# Now that the matrix is ready, with 1.0 on the diagonal,
	# it is time to convert the values into colors.

	color_matrix = np.zeros((n_params_per_atom, n_params_per_atom, 3))
	for m in range(n_params_per_atom):
		for n in range(n_params_per_atom):
			val   = correlation_matrix[m][n]
			r, g, b = 1.0, 1.0, 1.0
			if val > 0.0:
				r -= val
				b -= val
			else:
				g += val
				b += val

			if m == n:
				r, g, b = 0.0, 0.0, 0.0
			color_matrix[m][n][0] = r
			color_matrix[m][n][1] = g
			color_matrix[m][n][2] = b

	# Now that we have a color matric, graph it.
	fig, ax = plt.subplots()
	pl = ax.imshow(color_matrix, interpolation='none')

	ax.set_xticks(list(range(len(axis_labels))))
	ax.set_yticks(list(range(len(axis_labels))))
	ax.set_xticklabels(axis_labels)
	ax.set_yticklabels(axis_labels)

	# for m in range(correlation_matrix.shape[0]):
	# 	for n in range(correlation_matrix.shape[1]):
	# 		ax.text(n, m, '%i'%(int(round(100*correlation_matrix[m][n]))), ha="center", va="center", color="black", fontsize=6)

	
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(4.0)

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(4.0)

	if abs_display:
		title = "Correlation Matrix (absolute correlation)"
	else:
		title = "Correlation Matrix"

	fig.subplots_adjust(bottom=0.2)

	plt.title(title)

	plt.savefig(output_file, dpi=250)

if __name__ == '__main__':
	# This program takes 4 arguments.
	#     1) The json file containins the correlations.
	#     2) The path to write the final PNG to.
	#     3) Final image resolution in the form 1920x1080
	#     4) y or n, whether or not to display abs(correlation)

	if len(sys.argv) != 5:
		eprint("This program takes 4 arguments.")
		sys.exit(1)

	correlation_file = sys.argv[1]
	output_file      = sys.argv[2]
	x_res            = int(sys.argv[3].split('x')[0])
	y_res            = int(sys.argv[3].split('x')[1])
	abs_display      = sys.argv[4] == 'y'

	
	f = open(correlation_file, 'r')
	results = json.loads(f.read())
	f.close()

	GenHeatmap(results, output_file, abs_display)

	