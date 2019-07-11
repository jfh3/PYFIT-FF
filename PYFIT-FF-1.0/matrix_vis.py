import matplotlib.pyplot as plt
import numpy             as np
import code
import argparse
import os
import time
import sys
import json

from sys import path
sys.path.append('scripts')

import tl

def parse_slice(s):
	l, r = s.split(':')

	return slice(
		start=(int(l) if l != '' else None),
		stop =(int(r) if r != '' else None)
	)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Creates a plot that displays a matrix as a grid of " +
		"colors, with options for the coloration, axis labels and slice to " +
		"display in the plot."
	)

	parser.add_argument(
		'-m', '--matrix-file', dest='matrix_path', type=str, required=True,
		help='The matrix file to plot. See tl.py -> ndarray_to_file for format.'
	)

	parser.add_argument(
		'-l', '--label-file', dest='label_path', type=str, default='',
		help='A json file containing axis labels.'
	)

	parser.add_argument(
		'-s', '--slice', dest='slice', type=str, nargs=2, 
		metavar=('YSLICE', 'XSLICE'), default=[':', ':'],
		help='The Y and X slices of the data to display.'
	)

	parser.add_argument(
		'-f', '--save-file', dest='save_path', type=str, default='',
		help='The location to save the rendered file into.'
	)

	parser.add_argument(
		'-w', '--positive-color', dest='positive_color', type=int, nargs=3,
		metavar=('R', 'G', 'B'), default=[0, 255, 0],
		help='The color to display for positive values.'
	)

	parser.add_argument(
		'-e', '--negative-color', dest='negative_color', type=int, nargs=3,
		metavar=('R', 'G', 'B'), default=[255, 0, 0],
		help='The color to display for negative values.'
	)

	parser.add_argument(
		'-d', '--display-backend', dest='display_backend', type=str, default='TkAgg',
		help='The matplotlib backend to use when displaying the image.'
	)

	parser.add_argument(
		'-b', '--save-backend', dest='save_backend', type=str, default='Agg',
		help='The matplotlib backend to use when saving the image.'
	)

	parser.add_argument(
		'-p', '--save-dpi', dest='save_dpi', type=int, default=300,
		help='The dpi to use when saving the image.'
	)

	parser.add_argument(
		'-q', '--quiet-mode', dest='quiet', action='store_true',
		help='Don\'t display the graph.'
	)

	args = parser.parse_args()

	if args.quiet and args.save_path == '':
		print("When running in quiet mode, a save destination must be specified.")
		parser.print_help()
		exit(1)

	try:
		matrix = tl.ndarray_from_file(args.matrix_path)
	except:
		print("Failed to load the matrix file.")
		exit(1)

	if matrix.ndim != 2:
		print("Cannot display an object with ndim != 2.")
		exit(1)

	# Parse the slices into actual slice objects.
	try:
		yslice = parse_slice(args.slice[0])
		xslice = parse_slice(args.slice[1])
	except:
		print("Invalid slice format.")
		exit(1)

	# If axis labels were specified, load them.
	if args.label_path != '':
		try:
			with open(args.label_path, 'r') as file:
				labels = json.loads(file.read())
		except:
			print("Couldn't open or parse label file.")
			exit(1)

		try:
			xlabels = labels['x']
			ylabels = labels['y']
		except:
			print("Label file had invalid contents.")
			exit(1)
	else:
		xlabels = None
		ylabels = None

	# Now we process the data into a color matrix.
	






