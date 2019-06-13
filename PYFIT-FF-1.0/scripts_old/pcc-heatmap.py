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
	
	results_file      = 'pcc_calc_output/results-10-r0-sigma-0.25.json'

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
			label = "$P_" + str(l) + r"\;\;r = " + str(r) + "$"
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

		correlation_matrix[l_idx][r_idx]      = np.abs(pcc)
		correlation_matrix[r_idx][l_idx]      = np.abs(pcc)
		correlation_transformed[l_idx][r_idx] = np.abs(pcc_t)
		correlation_transformed[r_idx][l_idx] = np.abs(pcc_t)

	rmse = []
	total   = 0.0
	total_n = 0.0
	for m in range(n_params_per_atom):
		sum_sqr = 0.0
		n_sqr   = n_params_per_atom
		for n in range(n_params_per_atom):
			if m != n:
				sum_sqr += correlation_matrix[m][n]**2
				total   += correlation_matrix[m][n]**2
				total_n += 1
		sum_sqr /= n_sqr
		rmse.append(np.sqrt(sum_sqr))

	print(np.sqrt(total / total_n))

	for idx, r in enumerate(rmse):
		print('%2i : %f'%(idx, r))

	# f = open('plot3', 'w')
	# f.write(' '.join([str(i) for i in rmse]))
	# f.close()

	plt.scatter(range(n_params_per_atom), rmse)
	plt.show()


	for mn in range(n_params_per_atom):
		correlation_matrix[mn][mn] = 1.0

	# for m in range(24, 36):
	# 	for n in range(n_params_per_atom):
	# 		correlation_matrix[m][n] = 0.0
	# 		correlation_transformed[m][n] = 0.0

	# for m in range(n_params_per_atom):
	# 	for n in range(24, 36):
	# 		correlation_matrix[m][n] = 0.0
	# 		correlation_transformed[m][n] = 0.0

	# for mn in range(24, 36):
	# 	correlation_matrix[mn][mn] = 0.0
	# 	correlation_transformed[mn][mn] = 0.0

	# sum_sqr = 0.0
	# n_pcc       = 0


	# for m in range(n_params_per_atom):
	# 	for n in range(n_params_per_atom):
	# 		pcc = correlation_matrix[m][n]
	# 		sum_sqr += pcc**2
	# 		if pcc != 0.0:
	# 			n_pcc += 1

	# print(n)
	# print(np.sqrt(sum_sqr / n_pcc))

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
					ax.text(n, m, '%i'%(int(round(100*data_slice[m][n]))), ha="center", va="center", color="black", fontsize=8)
		
		def format_display(**kwargs):
			i = kwargs['i']
			j = kwargs['j']
			formatted =  ''
			formatted += "X Parameter: %s\n"%x_labels[j]
			formatted += "Y Parameter: %s\n"%y_labels[i]
			formatted += r"$\rho_{X,\;Y} = $" + '%1.3f'%(data_slice[i][j])
			return formatted

		datacursor(formatter=format_display)
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
	
	

	
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	fig.tight_layout()

	if label_values:
		if cube:
			title = r"$\rho_{X,\;Y}$" + " for all combinations of structural parameters (color cubed)"
		elif square:
			title = r"$\rho_{X,\;Y}$" + " for all combinations of structural parameters (color squared)"
		else:
			title = r"$\rho_{X,\;Y}$" + " for all combinations of structural parameters"
	else:
		if cube:
			title = r"$\rho_{X,\;Y}^3$" + " for all combinations of structural parameters"
		elif square:
			title = r"$\rho_{X,\;Y}^2$" + " for all combinations of structural parameters"
		else:
			title = r"$\rho_{X,\;Y}$" + " for all combinations of structural parameters"

	plt.title(title)

	plt.show()