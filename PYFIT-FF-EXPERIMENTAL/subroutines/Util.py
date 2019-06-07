from Config   import *
from datetime import datetime
import numpy as np
import sys
import atexit

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
def init(lp = None):
	global log_file
	global indent
	if lp != None:
		log_file  = open(lp, 'w')
	else:
		log_file  = open(LOG_PATH, 'w')
	indent    = 0

# This ensures that all log file information will write
# in the event of a crash. (hopefully)
@atexit.register
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
	#print(string)

def log_indent():
	global indent
	indent += 1

def log_unindent():
	global indent
	indent -= 1

class ProgressBar:
	def __init__(self, prefix, prefix_width, total, update_every = 10, border=['', ''], space_char='░', fill_char='█'):
		self.prefix        = prefix
		self.prefix_width  = prefix_width
		self.update_every  = update_every
		self.total         = total
		self.current       = 0.0
		self.fill_char     = fill_char
		self.space_char    = space_char
		self.border        = border
		self.start_time    = datetime.now()
		self.time_per_unit = 0.01
		self.update_count  = 0
		self.display()

	def update(self, value):
		self.current      =  value
		self.update_count += 1

		if self.update_count % self.update_every == 0 or self.update_count == 1:
			self.time_per_unit = (datetime.now() - self.start_time).seconds / self.current
			self.display()

	def finish(self):
		self.current = self.total
		total_time   = (datetime.now() - self.start_time).seconds
		seconds      = int(total_time % 60)
		minutes      = int(np.floor(total_time / 60))
		time         = ' (%02i:%02i elapsed)'%(minutes, seconds)
		self.display(_end='')
		print(time, end='')
		print('\n', end='')

	# This function returns a tuple with the first member being the
	# percentage to display and the second number being the number
	# of ticks to draw in the progress bar.
	def get_display(self):
		percentage = (self.current / self.total) * 100
		ticks      = int(np.floor((self.current / self.total) * 50))
		return (ticks, percentage)

	def display(self, _end='\r'):
		ticks, percentage = self.get_display()
		fill   = self.fill_char  * ticks       # 0 - 50 fill characters
		space  = self.space_char * (50 - ticks) # 0 - 50 space characters
		disp   = '%' + '%05.2f'%(percentage)
		prefix = self.prefix + (' ' * (self.prefix_width - len(self.prefix)))


		# This is the only consistent way to clear the current line.
		# Using a \r character at the end of the line only works for
		# some environments. Odds are that this will not work on windows
		# but who cares.
		sys.stdout.write("\033[K")
		print(prefix + self.border[0] + fill + space + self.border[1] + ' ' + disp, end=_end)