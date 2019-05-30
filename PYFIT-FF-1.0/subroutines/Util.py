from Config   import *
from datetime import datetime
import numpy as np

def GetLineCells(line_string):
	return [i for i in line_string.split(' ') if i != '' and not i.isspace()]

def NormalizeLineEndings(string):
	out = ''
	skip = False
	for i, c in enumerate(string):
		if skip:
			skip = False
		else:
			if c == '\r':
				if i != len(string) - 1:
					if string[i + 1] == '\n':
						out.append('\n')
						skip = True
			else:
				out.append(c)
	return out

def LogConfiguration():
	log("---------- Begin Configuration ----------")
	f = open(CONFIG_FNAME, 'r')
	lines = f.read().split('\n')

	# Take all lines that aren't whitespace and don't start 
	# with '#'. Split them up at the equal sign and align
	# all of the equal signs for readability.

	max_length = 0
	cleaned    = []
	for line in lines:
		if not line.isspace() and line != '' and line.lstrip()[0] != '#':
			cells = line.lstrip().rstrip().split('=')
			if len(cells[0]) > max_length:
				max_length = len(cells[0])
			cleaned.append(cells)

	# Now rebuild a list with all of the equal signs aligned
	new_lines = []
	for line in cleaned:
		if len(line[0]) < max_length:
			new_lines.append(line[0] + (' ' * (max_length - len(line[0]))) + ' = ' + line[1])
		else:
			new_lines.append(line[0] + ' = ' + line[1])

	log('\n'.join(new_lines)) 

	log("----------- End Configuration -----------\n")

# -------------------------------------
# Initialization and Logging
# -------------------------------------

log_file  = open(LOG_PATH, 'w')
indent    = 0

def cleanup():
	log_file.close()

def log(string, endl = '\n'):
	if '\n' in string.rstrip():
		# Split it up and apply the intended indent.
		lines = string.split('\n')
		new_lines = []
		for line in lines:
			log_file.write(('\t' * indent) + line + endl)
	else:
		log_file.write(('\t' * indent) + string + endl)

def log_indent():
	global indent
	indent += 1

def log_unindent():
	global indent
	indent -= 1