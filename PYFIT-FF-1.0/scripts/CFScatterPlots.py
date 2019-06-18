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

def GenCFScatterPlots(results, output_dir):
	for correlation in results["data"]:

		fig, ax = plt.subplots()

		current_file = output_dir + '%i_vs_E.png'%(correlation["param"]["idx"])

		horizontal_axis_label  = '$G_{%i}$  '%correlation["param"]["idx"]
		horizontal_axis_label += '$P_{%i}$  '%correlation["param"]["l"]
		horizontal_axis_label += '$r_0 = %f$'%correlation["param"]["r0"]

		vertical_axis_label  = '$E$'

		title = '$G_{%i}$ vs. $E$ ($\\rho_{X, Y} = \\;$%1.2f)'%(correlation["param"]["idx"], correlation["pcc"])

		x_points = correlation["inputs"]
		y_points = correlation["outputs"]

		ax.scatter(x_points, y_points, s=0.1, alpha=0.25)
		plt.title(title)
		plt.xlabel(horizontal_axis_label)
		plt.ylabel(vertical_axis_label)
		plt.savefig(current_file, dpi=250)

		plt.close(fig)

if __name__ == '__main__':
	# This program takes 3 arguments.
	#     1) The json file containins the correlations.
	#     2) The directory to put the files in.
	#     3) The portion of data points to actually graph.

	if len(sys.argv) != 4:
		eprint("This program takes 3 arguments.")
		sys.exit(1)

	correlation_file = sys.argv[1]
	output_dir       = sys.argv[2]

	
	f = open(correlation_file, 'r')
	results = json.loads(f.read())
	f.close()

	GenCFScatterPlots(results, output_dir)

	