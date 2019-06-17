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

def GenCFHeatmap(correlation_results, abs_display, output_file):
	correlation_matrix = np.zeros((len(correlation_results), len(correlation_results[0]["data"])))

	horizontal_axis_labels = []
	for param in correlation_results[0]["data"]:
		r = param["param"]["r0"]
		l = param["param"]["l"]
		
		label = "$P_" + str(l) + r"\;\;r = " + str(r) + "$"
		horizontal_axis_labels.append(label)

	vertical_axis_labels = ["IC-%02i"%i for i in range(len(correlation_results))]

	
	for row in range(len(correlation_results)):
		result = correlation_results[row]
		# Construct an easy to graph matrix.
		for idx, coeff in zip(range(len(result["data"])), result["data"]):
			pcc = coeff["pcc"]

			if np.abs(pcc) > 1.0:
				eprint("ERROR: One of the coefficients was outside of [-1, 1]")
				sys.exit(1)

			if abs_display:
				correlation_matrix[row][idx] = np.abs(pcc)
			else:
				correlation_matrix[row][idx] = pcc

	# Now that the matrix is ready, with 1.0 on the diagonal,
	# it is time to convert the values into colors.

	color_matrix = np.zeros((correlation_matrix.shape[0], correlation_matrix.shape[1], 3))
	for m in range(correlation_matrix.shape[0]):
		for n in range(correlation_matrix.shape[1]):
			val = correlation_matrix[m][n]
			r, g, b = 1.0, 1.0, 1.0
			if val > 0.0:
				r -= val
				b -= val
			else:
				g += val
				b += val

			color_matrix[m][n][0] = r
			color_matrix[m][n][1] = g
			color_matrix[m][n][2] = b

	# Now that we have a color matric, graph it.
	fig, ax = plt.subplots()
	pl = ax.imshow(color_matrix, interpolation='none')

	ax.set_xticks(list(range(len(horizontal_axis_labels))))
	ax.set_yticks(list(range(len(vertical_axis_labels))))
	ax.set_xticklabels(horizontal_axis_labels)
	ax.set_yticklabels(vertical_axis_labels)

	# for m in range(correlation_matrix.shape[0]):
	# 	for n in range(correlation_matrix.shape[1]):
	# 		ax.text(n, m, '%i'%(int(round(100*correlation_matrix[m][n]))), ha="center", va="center", color="black", fontsize=6)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(6.5)

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(6.5)
	
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	if abs_display:
		title = "Correlation Matrix (absolute correlation)"
	else:
		title = "Correlation Matrix"

	fig.subplots_adjust(bottom=0.2)

	plt.title(title)

	plt.savefig(output_file, dpi=250)

if __name__ == '__main__':
	# This program takes a variable number of arguments.
	#     1) The path to write the png file to.
	#     2) y or n, whether or not to display abs(correlation)
	#     *) The correlation files to load data from.

	if len(sys.argv) < 4:
		eprint("This program takes at least 4 arguments.")
		sys.exit(1)

	output_file = sys.argv[1]
	abs_display = sys.argv[2] == 'y'
	
	correlation_files = sys.argv[3:]
	
	
	correlation_results = []
	
	for file in correlation_files:
		f = open(file, 'r')
		correlation_results.append(json.loads(f.read()))
		f.close()

	GenCFHeatmap(correlation_results, abs_display, output_file)
	