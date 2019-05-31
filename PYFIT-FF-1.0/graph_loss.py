import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	fname = sys.argv[1]
	file  = open(fname, 'r')
	raw   = file.read()
	