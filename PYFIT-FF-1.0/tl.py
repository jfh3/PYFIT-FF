import os
import sys
import numpy as np
import struct

# Find all files matching either an exact name, or an extension,
# within the given directory. This function will return two arrays,
# the first will contain the location of each file, as an absolute
# path. The second will contain the actual file contents, as a string.
# If only an extension is specified, each element of both returned 
# arrays will be a list with one element for each matching file
# in each directory. This function will stop going deeper into 
# each directory when it finds matching files. The third return
# value is a dictionary containing information about what was 
# found during the search.
def walk_dir_recursive(directory, max_depth, **kwargs):
	valid_kwargs = ['extension', 'name']
	for k in kwargs.keys():
		if k not in valid_kwargs:
			raise ValueError("Invalid Keyword Argument: %s"%k)

	depth          = max_depth
	extension_mode = 'extension' in kwargs.keys()


	directory = os.path.abspath(os.path.expanduser(directory))

	if not extension_mode:
		if 'name' not in kwargs.keys():
			msg  = "You must specify either an extension to search for" 
			msg += " or a file name to search for."
			raise ValueError(msg)
	elif 'name' in kwargs.keys():
		raise ValueError("Cannot specify both 'extension' and 'name'.")

	if extension_mode:
		target = kwargs['extension']
		if target[0] == '.':
			target = target[1:]
	else:
		target = kwargs['name']

	if isinstance(target, list):
		target = [t.strip() for t in target]
	else:
		target = target.strip()

	stats = {
		'dirs_searched'           : 0,
		'top_level_dirs_searched' : 0,
		'max_depth_searched'      : 1,
		'matches'                 : 0,
		'read_failures'           : 0
	}

	locations     = []
	file_contents = []

	for item in os.listdir(directory):
		proper_item = os.path.join(directory, item)
		if os.path.isdir(proper_item):
			paths, contents, status = _inner_walk_dir_recursive(
				proper_item, max_depth, 1,
				extension_mode, target
			)



			stats['top_level_dirs_searched'] += 1
			stats['max_depth_searched']       = max([
				stats['max_depth_searched'], 
				status['max_depth_searched']
			])

			stats['dirs_searched'] += status['dirs_searched']
			stats['matches']       += status['matches']
			stats['read_failures'] += status['read_failures']

			if [paths, contents] == [None, None]:
				# This directory doesn't have any matches.
				# Do nothing.
				pass
			else:
				locations.append(paths)
				file_contents.append(contents)

	return locations, file_contents, stats

def _inner_walk_dir_recursive(directory, max_depth, depth, extension_mode, target):
	if depth > max_depth:
		return None, None, {
			'max_depth_searched' : depth,
			'dirs_searched'      : 0,
			'matches'            : 0,
			'read_failures'      : 0
		}

	stats = {
		'max_depth_searched' : depth,
		'dirs_searched'      : 1,
		'matches'            : 0,
		'read_failures'      : 0
	}

	subdirs = []
	files   = []
	for item in os.listdir(directory):
		item_proper = os.path.join(directory, item)
		if os.path.isdir(item_proper):
			subdirs.append(item_proper)
		elif os.path.isfile(item_proper):
			files.append(item_proper)

	matches = []
	for file in files:
		matchfile = os.path.split(file)[-1]
		if extension_mode:
			check = matchfile.split('.')[-1].strip()
			if check == target:
				matches.append(file)
		else:
			if isinstance(target, list):
				if matchfile.strip() in target:
					matches.append(file)
			else:
				if matchfile.strip() == target:
					matches.append(file)
					break

	if len(matches) != 0:
		stats['matches'] = len(matches)

		if extension_mode or isinstance(target, list):
			locations = []
			contents  = []
			
			for match in matches:
				try:
					with open(match, 'r') as f:
						raw       = f.read()
						full_path = os.path.abspath(match)
				except:
					stats['read_failures'] += 1
					continue

				locations.append(full_path)
				contents.append(raw)


			return locations, contents, stats
		else:
			try:
				with open(matches[0], 'r') as f:
					raw       = f.read()
					full_path = os.path.abspath(matches[0])
					return full_path, raw, stats
			except:
				stats['read_failures'] += 1

	# If we get to here, we didn't find anything, run through
	# all subdirectories.
	if len(subdirs) != 0:
		for subdir in subdirs:
			item_proper = os.path.join(directory, subdir)
			loc, raw, stat = _inner_walk_dir_recursive(
				item_proper, max_depth, depth + 1,
				extension_mode, target
			)

			stats['max_depth_searched'] = max([
				stats['max_depth_searched'], 
				stat['max_depth_searched']
			])

			stats['matches'] += stat['matches']

			stats['dirs_searched'] += stat['dirs_searched']
			stats['read_failures'] += stat['read_failures']

			if [loc, raw] != [None, None]:
				return loc, raw, stats

		return None, None, stats
	else:
		return None, None, stats


# Writes the given matrix to a file in a format that should
# be very fast to read. If you pass a serialize function, it
# will be called on every element of the matrix to write it 
# to the file. Every element needs to be the same number of 
# bytes though.
def mat_to_file(fpath, contents):
	with open(fpath, 'wb') as file:
		file.write(contents.shape[0].to_bytes(8, byteorder='little'))
		file.write(contents.shape[1].to_bytes(8, byteorder='little'))
		file.write(contents.tobytes())

def mat_from_file(fpath, dtype):
	with open(fpath, 'rb') as file:
		data = file.read()
		rows = int.from_bytes(data[0:8],  byteorder='little')
		cols = int.from_bytes(data[8:16], byteorder='little')
		arr  = np.frombuffer(data[16:], dtype=dtype).reshape((rows, cols))

	return arr

	

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Type a python expression in double quotes to test it.")
	else:
		print(*eval(sys.argv[1]))