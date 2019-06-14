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
	graph_ratio      = float(sys.argv[3])

	
	f = open(correlation_file, 'r')
	results = json.loads(f.read())
	f.close()

	for correlation in results["coefficients"]:

		fig, ax = plt.subplots()

		current_file = output_dir + '%i_vs_%i.png'%(correlation["param0"]["idx"], correlation["param1"]["idx"])

		horizontal_axis_label  = '$G_{%i}$  '%correlation["param0"]["idx"]
		horizontal_axis_label += '$P_{%i}$  '%correlation["param0"]["l"]
		horizontal_axis_label += '$r_0 = %f$'%correlation["param0"]["r0"]

		vertical_axis_label  = '$G_{%i}$  '%correlation["param1"]["idx"]
		vertical_axis_label += '$P_{%i}$  '%correlation["param1"]["l"]
		vertical_axis_label += '$r_0 = %f$'%correlation["param1"]["r0"]

		title = '$G_{%i}$ vs. $G_{%i}$ ($\\rho_{X, Y} = \\;$%1.2f)'%(correlation["param0"]["idx"], correlation["param1"]["idx"], correlation["pcc"])

		x_points = np.random.choice([p[0] for p in correlation["data"]], int(round(graph_ratio*len(correlation["data"]))))
		y_points = np.random.choice([p[1] for p in correlation["data"]], int(round(graph_ratio*len(correlation["data"]))))

		ax.scatter(x_points, y_points, s=0.1)
		plt.title(title)
		plt.xlabel(horizontal_axis_label)
		plt.ylabel(vertical_axis_label)
		plt.savefig(current_file, dpi=250)

		plt.close(fig)