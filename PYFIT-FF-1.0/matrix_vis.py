import matplotlib
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
		(int(l) if l != '' else None),
		(int(r) if r != '' else None)
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
		'--positive-color', dest='positive_color', type=float, nargs=3,
		metavar=('R', 'G', 'B'), default=[0., 1., 0.],
		help='The color to display for positive values. (float [0.0, 1.0])'
	)

	parser.add_argument(
		'--negative-color', dest='negative_color', type=float, nargs=3,
		metavar=('R', 'G', 'B'), default=[1., 0., 0.],
		help='The color to display for negative values. (float [0.0, 1.0])'
	)

	parser.add_argument(
		'-i', '--min-val', dest='min_val', type=float, default=-1e9,
		help='Treat this as the minimum value when scaling colors.'
	)

	parser.add_argument(
		'-a', '--max-val', dest='max_val', type=float, default=1e9,
		help='Treat this as the maximum value when scaling colors.'
	)

	parser.add_argument(
		'-d', '--display-backend', dest='display_backend', type=str, default='',
		help='The matplotlib backend to use when displaying the image.'
	)

	parser.add_argument(
		'-b', '--save-backend', dest='save_backend', type=str, default='',
		help='The matplotlib backend to use when saving the image.'
	)

	parser.add_argument(
		'-t', '--title', dest='title', type=str, default='',
		help='The title for the plot.'
	)

	parser.add_argument(
		'--number-format', dest='number_format', type=str, default='',
		help='A Python expression describing how to format a number as a ' +
		'string for display in the grid. Use \'v\' as the value to format ' +
		'in this expression'
	)

	parser.add_argument(
		'--label-format', dest='label_format', type=str, default='',
		help='A Python expression describing how to format labels. ' +
		'Use \'v\' as the value to format in this expression'
	)

	parser.add_argument(
		'--value-mod', dest='value_mod', type=str, default='',
		help='A Python expression that modifies each value before display. ' +
		'Use \'v\' as the value to format in this expression'
	)

	parser.add_argument(
		'--number-grid', dest='number', action='store_true',
		help='Number the squares in the plot with their value.'
	)

	parser.add_argument(
		'-z', '--number-font', dest='number_font', type=float, default=6,
		help='Font size for the square numbers.'
	)

	parser.add_argument(
		'-x', '--x-font', dest='x_font', type=float, default=4,
		help='Font size for the x axis.'
	)

	parser.add_argument(
		'-y', '--y-font', dest='y_font', type=float, default=4,
		help='Font size for the y axis.'
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

	if not args.quiet and args.save_path != '':
		print("Cannot use quiet mode and save mode at the same time.")
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

			if args.label_format != '':
				xlabels = [eval(args.label_format) for v in xlabels]
				ylabels = [eval(args.label_format) for v in ylabels]
		except:
			print("Label file had invalid contents.")
			exit(1)
	else:
		xlabels = None
		ylabels = None

	# Now we process the data into a color matrix.
	if args.min_val == -1e9:
		args.min_val = matrix.min()

	if args.max_val == 1e9:
		args.max_val = matrix.max()

	if args.value_mod != '':
		new_array = np.zeros((matrix.shape[0], matrix.shape[1]))
		for i in range(matrix.shape[0]):
			for j in range(matrix.shape[1]):
				v = matrix[i][j]
				new_array[i, j] = eval(args.value_mod)
		matrix = new_array

	color_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 3))

	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			val = matrix[i][j]

			# This should be a variation of args.positive_color if its
			# in the top half of the valid range. It should be a variation
			# of args.negative_color if its in the bottom half.
			center = np.array([args.max_val, args.min_val]).mean()

			positive = val >= center

			# Now we interpolate from white to the appropriate color, 
			# based on the distance from the center value.

			distance = val - center

			color = args.positive_color if positive else args.negative_color

			if not positive:
				distance = -distance

			def interp(v, c):
				return 1.0 - (1.0 - c)*(v / ((args.max_val - args.min_val) / 2))

			color_matrix[i][j] = [
				interp(distance, color[0]),
				interp(distance, color[1]),
				interp(distance, color[2])
			]

	if args.display_backend != '':
		matplotlib.use(args.display_backend)

	if args.save_backend != '':
			matplotlib.use(args.save_backend)
	
	fig, ax = plt.subplots()
	pl = ax.imshow(color_matrix[yslice, xslice], interpolation='none')

	if xlabels is not None:
		ylabels = ylabels[yslice]
		xlabels = xlabels[xslice]

		ax.set_xticks(list(range(len(xlabels))))
		ax.set_yticks(list(range(len(ylabels))))

		ax.set_xticklabels(xlabels)
		ax.set_yticklabels(ylabels)

	if args.number:
		for m in range(matrix[yslice, xslice].shape[0]):
			for n in range(matrix[yslice, xslice].shape[1]):
				v = matrix[m][n]
				if args.number_format != '':
					v = eval(args.number_format)

				ax.text(
					n, m, v, ha="center", va="center", 
					color="black", fontsize=args.number_font
				)


	plt.setp(
		ax.get_xticklabels(), rotation=45, 
		ha="right", rotation_mode="anchor"
	)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(args.x_font)

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(args.y_font)

	plt.title(args.title)
	plt.tight_layout()
	

	if not args.quiet:
		plt.show()
	else:
		plt.savefig(args.save_path, dpi=args.save_dpi)


