import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import json
from   time          import time
from   datetime      import datetime
from   mpldatacursor import datacursor

import warnings
warnings.filterwarnings("ignore")

def parse_slice(s):
	l, r = s.split(':')
	if l == '' and r == '':
		return slice(None, None)
	elif l == '':
		return slice(None, int(r))
	elif r == '':
		return slice(int(l), None)
	else:
		return slice(int(l), int(r))

if __name__ == '__main__':
	

	results_file      = 'pcc_calc_output/results.json'

	sliced = False
	if len(sys.argv) >= 3:
		sliced = True
		l_slice = parse_slice(sys.argv[1])
		r_slice = parse_slice(sys.argv[2])

	square = False
	cube   = False
	if len(sys.argv) >= 4:
		square = sys.argv[3].lower() == 'square'
		cube   = sys.argv[3].lower() == 'cube'
		

	label_values = False
	if len(sys.argv) >= 5:
		label_values = sys.argv[4].lower() == 'label'
	
	if square:
		print("squaring all values")

	if cube:
		print("cubing all values")

	if label_values:
		print("labelling all values")

	f = open(results_file, 'r')
	results = json.loads(f.read())
	f.close()

	n_params_per_atom       = results["n_params_per_atom"]
	correlation_matrix      = np.zeros((n_params_per_atom, n_params_per_atom))
	correlation_transformed = np.zeros((n_params_per_atom, n_params_per_atom))

	axis_labels = []
	for l in results["legendre_polynomials"]:
		for r in results["r_0_values"]:
			label = "$P_" + str(l) + r"\;r = " + str(r) + "$"
			axis_labels.append(label)

	# Construct an easy to graph matrix.
	for coeff in results["coefficients"]:
		l_idx = coeff["param0"]["idx"]
		r_idx = coeff["param1"]["idx"]
		pcc   = coeff["pcc"]

		if square:
			_pcc = pcc**2
			if pcc < 0.0:
				pcc_t = -_pcc
			else:
				pcc_t = _pcc
		elif cube:
			pcc_t = pcc**3
		else:
			pcc_t = pcc

		if np.abs(pcc) > 1.0:
			print("ERROR: One of the coefficients was outside of [-1, 1]")
			exit()

		correlation_matrix[l_idx][r_idx]      = pcc
		correlation_matrix[r_idx][l_idx]      = pcc
		correlation_transformed[l_idx][r_idx] = pcc_t
		correlation_transformed[r_idx][l_idx] = pcc_t

	for mn in range(n_params_per_atom):
		correlation_matrix[mn][mn] = 1.0

	# Now that the matrix is ready, with 1.0 on the diagonal,
	# it is time to convert the values into colors.

	color_matrix = np.zeros((n_params_per_atom, n_params_per_atom, 3))
	for m in range(n_params_per_atom):
		for n in range(n_params_per_atom):
			val   = correlation_transformed[m][n]
			r, g, b = 1.0, 1.0, 1.0
			if val > 0.0:
				r -= val
				b -= val
			else:
				g += val
				b += val

			if m == n:
				r, g, b = 0.0, 0.898, 1.0
			color_matrix[m][n][0] = r
			color_matrix[m][n][1] = g
			color_matrix[m][n][2] = b

	# Now that we have a color matric, graph it.
	fig, ax = plt.subplots()
	if sliced:
		plot_slice = color_matrix[l_slice, r_slice]
		data_slice = correlation_matrix[l_slice, r_slice]
		pl         = ax.imshow(plot_slice, interpolation='none')

		x_labels = axis_labels[r_slice]
		y_labels = axis_labels[l_slice]

		ax.set_xticks(list(range(len(x_labels))))
		ax.set_yticks(list(range(len(y_labels))))
		ax.set_xticklabels(x_labels)
		ax.set_yticklabels(y_labels)

		if label_values:
			for m in range(data_slice.shape[0]):
				for n in range(data_slice.shape[1]):
					ax.text(m, n, '%i'%(int(round(100*data_slice[m][n]))), ha="center", va="center", color="black", fontsize=8)
	else:
		pl = ax.imshow(color_matrix, interpolation='none')
		ax.set_xticks(list(range(n_params_per_atom)))
		ax.set_yticks(list(range(n_params_per_atom)))
		ax.set_xticklabels(axis_labels)
		ax.set_yticklabels(axis_labels)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(8)

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(8)
	
	datacursor(display='single')
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	fig.tight_layout()

	plt.show()