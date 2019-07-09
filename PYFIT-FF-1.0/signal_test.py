import signal
import time
import argparse


def stop(n, f):
	print("Stopping ????")

def start(n, f):
	print("Starting ????")

if __name__ == '__main__':


signal.signal(signal.SIGSTSTP, stop)
signal.signal(signal.SIGCONT, start)

for i in range(10000000):
	print("Iteration: %010i"%i)
	time.sleep(0.1)