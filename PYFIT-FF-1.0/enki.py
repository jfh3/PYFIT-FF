import matplotlib.pyplot as plt
import numpy             as np
import code
import argparse
import os
import time
import sys
import subprocess
import getpass
import json

def run(cmdline):
	process       = subprocess.Popen(cmdline, shell=True)
	output, error = process.communicate()

	if error != '' and error != None:
		raise Exception("Command Error: %s"%error)

def run_enki(cmdline, pwd, wd=None):
	if wd is not None:
		run('sshpass -p "%s" ssh -t ajr6@enki.nist.gov \"cd %s && %s\"'%(
			pwd, wd, cmdline.replace('"', '\\"').replace('~', '\\~')
		))
	else:
		run('sshpass -p "%s" ssh -t ajr6@enki.nist.gov \"%s\"'%(
			pwd, cmdline.replace('"', '\\"').replace('~', '\\~')
		))

def validate_args(args, parser):
	if args.command == []:
		print("No command specified.")
		parser.print_help()
		exit(1)

	# Before we do anything, make sure that the args make sense.
	if args.partition not in ['gpu', 'general', 'debug']:
		print("-p/--partition can only take the values gpu, general and debug.")
		parser.print_help()
		exit(1)

	if args.n_gpu != 0 and args.partition == 'general':
		if not args.no_warn:
			msg  = 'You should probably not use the general queue if you are using '
			msg += 'gpus. Specify --no-warn to run anyways.'
			print(msg)
			parser.print_help()
			exit(1)

	if args.n_gpu < 0:
		print('Negative gpu count specified.')
		parser.print_help()
		exit(1)

	if args.n_gpu != 0:
		if args.partition == 'gpu':
			if args.n_gpu > 24:
				print('Max gpu count for partition \'gpu\' is 24.')
				parser.print_help()
				exit(1)

		if args.partition == 'debug':
			if args.n_gpu > 8:
				print('Max gpu count for partition \'debug\' is 8.')
				parser.print_help()
				exit(1)

	if args.n_cores < 0:
		print('Negative core count specified.')
		parser.print_help()
		exit(1)

	if args.partition == 'gpu':
		if args.n_cores > 960:
			print('Max core count for partition \'gpu\' is 960.')
			parser.print_help()
			exit(1)

	if args.partition == 'debug':
		if args.n_cores > 320:
			print('Max core count for partition \'debug\' is 320.')
			parser.print_help()
			exit(1)

	# Try to ensure that the working directory is going to be valid and 
	# isn't somewhere that is going to be deleted automatically.
	path_split = [i for i in args.working_directory.split('/') if i != '']
	start      = path_split[0] + '/' + path_split[1]
	if start not in ['home/ajr6', 'wrk/ajr6']:
		if not args.no_warn:
			print("The working directory specified looks incorrect.")
			print("Specify --no-warn to run anyways.")
			parser.print_help()
			exit(1)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Quickly runs the given job on enki, with the given ' +
		' parameters. Will prompt for password.',
		epilog='Will create a slurm script, copy it to enki and execute it ' +
		'using ssh. An attempt will be made to parse the output of the sbatch ' +
		'command. If everything looks good, the job number will be reported. ' +
		'otherwise, the output will be printed and it will be up to you to ' + 
		'figure out if you need to restart the run.'
	)

	parser.add_argument(
		'-w', '--working-directory', dest='working_directory', type=str, 
		required=True, help='Absolute path, on enki to the directory to ' +
		'work in. Will be created if it does not exist.'
	)

	parser.add_argument(
		'-p', '--partition', dest='partition', type=str, required=True, 
		help='The partition to run the job in. (debug, gpu, general)'
	)

	parser.add_argument(
		'-j', '--job-name', dest='job_name', type=str, required=True, 
		help='What to name the job on enki.'
	)

	parser.add_argument(
		'-n', '--n-cores', dest='n_cores', type=int, required=True, 
		help='The number of cores to run the job with.'
	)

	parser.add_argument(
		'-t', '--time', dest='time', type=str, default='00:30:00', 
		help='The amount of time that should be allocated for the job. ' +
		'Default is 00:30:00.'
	)

	parser.add_argument(
		'-g', '--n-gpu', dest='n_gpu', type=int, default=0, 
		help='The number of gpus to run the job with. (default 0)'
	)

	parser.add_argument(
		'-y', '--copy', dest='copy', type=str, nargs='*', default=[],
		help='What to copy to the working directory in order to properly ' +
		'run the job. Specify \"./\" to copy the current directory.'
	)

	parser.add_argument(
		'-s', '--slurm-template', dest='slurm_template', type=str, 
		default='enki/enki_template.sh', 
		help='The file to use as a template sbatch script.'
	)

	parser.add_argument(
		'--no-warn', dest='no_warn', action='store_true', 
		help='Don\'t stop for a warning, just keep going.'
	)

	parser.add_argument(
		dest='command', type=str, nargs='*', default=[],
		help='The command line to run within the slurm script that will be ' +
		'generated and copied to enki.'
	)

	args = parser.parse_args()

	validate_args(args, parser)

	if args.working_directory[-1] != '/':
		args.working_directory += '/'

	# Now that the arguments are validated, it's time to ask for the password.

	password = getpass.getpass()

	print("Attempting to start ENKI job:")
	print("\tworking directory = %s"%(args.working_directory))
	print("\tpartition         = %s"%(args.partition))
	print("\tgpu count         = %s"%(args.n_gpu))
	print("\tcpu count         = %s"%(args.n_cores))
	print("\ttime              = %s"%(args.time))
	print("\ttemplate          = %s"%(args.slurm_template))
	print("\tscript copy       = %s"%(args.script_copy))
	print("\tcommand           = %s"%(' '.join(args.command)))


	# Now we create the working directory on enki and copy the necessary 
	# working files over.
	run_enki('mkdir %s'%args.working_directory, password)

	# Copy the working files over.
	if args.copy != []:
		if args.copy == ['./']:
			run('sshpass -p "%s" scp -r ./ ajr6@enki.nist.gov:%s'%(
				password, args.working_directory
			))
		else:
			for file in args.copy:
				run('sshpass -p "%s" scp %s ajr6@enki.nist.gov:%s'%(
					password, file, args.working_directory
				))

	# Generate an sbatch script and copy it to enki.

	try:
		with open(args.slurm_template, 'r') as file:
			template = file.read()
	except:
		print("Error reading template file.")
		exit(1)


	script = template.replace('{{{job_name}}}', str(args.job_name))
	script = script.replace('{{{n_gpu}}}',      str(args.n_gpu))
	script = script.replace('{{{n_cores}}}',    str(args.n_cores))
	script = script.replace('{{{partition}}}',  str(args.partition))
	script = script.replace('{{{time}}}',       str(args.time))
	script = script.replace('{{{command}}}', ' '.join(args.command))

	script_name = 'enki_run_script_tmp.sh'

	# Write the script to a temp file, copy it to enki and run it.
	try:
		with open(script_name, 'w') as file:
			file.write(script)
	except:
		print("Could not create the script file on local machine.")
		exit(1)


	run('sshpass -p "%s" scp %s ajr6@enki.nist.gov:%s'%(
		password, script_name, args.working_directory
	))

	# The file is now on enki, run it through ssh.

	script_path = args.working_directory + script_name

	run_enki('chmod +x %s'%script_path, password)

	run_enki('./%s'%script_name, password, wd=args.working_directory)



