import psutil
import sys
import time
import os


if __name__ == '__main__':
	args = [a.lower() for a in sys.argv[1:]]

	config = {}
	for arg in args:
		if arg == '--max-memory':
			config['max-memory'] = int(args[args.index('--max-memory') + 1])
		elif arg == '--interval':
			config['interval'] = float(args[args.index('--interval') + 1])

	if 'interval' not in config:
		config['interval'] = 0.25

	print("INTERVAL   = %f"%config['interval'])

	if 'max-memory' in config:
		print("MAX MEMORY = %i MB"%config['max-memory'])
	print("Monitoring . . . ")

	while True:
		try:
			time.sleep(config['interval'])
			for pid in psutil.pids():
				try:
					if 'max-memory' in config:
						process = psutil.Process(pid)
						mem     = process.memory_info().data / 1024 / 1024
						if mem >= config['max-memory']:
							print(mem)
							print("Killing Process %i . . . "%(pid))
							process.kill()
							print("Done . . . ")

				except:
					pass
		except KeyboardInterrupt:
			print("Exiting . . . ")
			break