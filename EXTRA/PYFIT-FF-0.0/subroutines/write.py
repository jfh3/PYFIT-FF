import os
from input_file import *  


def write_to_log(string):
	print(string)  #also write to std_out
	myfile = open(logfile, 'a')
	try:
		myfile.write('%s\n' % (string))
	finally:
   		myfile.close()
	return 


def write_input(in_array):
	[pot_type,wb_bnd,e_shift,NNlayers,param,ro,W,Gi_type,Gi_shift,tfunc,element,atomic_weight,Rc,Tc,sigma,Nro]=in_array
	write_to_log("# INPUT VARIABLES")
	write_to_log("# 	pot_type="+str(pot_type))
	write_to_log("# 	wb_bnd="+str(wb_bnd))
	write_to_log("# 	e_shift="+str(e_shift))
	write_to_log("# 	NNlayers="+str(NNlayers))
	write_to_log("# 	Activation function type="+str(tfunc)+' (0=sigmoid)(1=sigmoid-0.5)')
	write_to_log("# 	Number of fitting param (weights+bias)="+str(len(W)))
	write_to_log("# 	Gi ro locations="+str(ro))
	write_to_log("# 	Gi sigma="+str(sigma))
	write_to_log("# 	Gi_type="+str(Gi_type))
	write_to_log("# 	Gi_shift="+str(Gi_shift))
	write_to_log("# 	Rc (cutoff distance)="+str(Rc))
	write_to_log("# 	Tc (cutoff truncation range)="+str(Tc))


def write_batch(train_indices,batch_sids,structs):
	Ns1=0; Ns2=0
	for i3 in train_indices: 
		Ns1=Ns1+1		
		with open('output/FULL.dat', 'a') as out:
			out.write('%10f %10f  \n' % (structs[i3][3],structs[i3][1]/structs[i3][0]))
	for j1 in range(0,len(batch_sids)): 
		for str_i in batch_sids[j1]:
			Ns2=Ns2+1		
			with open('output/BATCH-'+str(j1)+'.dat', 'a') as out:
				out.write('%10f %10f  \n' % (structs[str_i][3],structs[str_i][1]/structs[str_i][0]))
#	if(Ns1!=Ns2):
#		print("LOST STRUCTURE (EXITING):"); exit()


def write_NN(W,NNlayers,ro,param,file_name='NN.dat'):
	[Gi_type,Gi_shift,tfunc,element,atomic_weight,Rc,Tc,sigma,Nro]=param

	myfile = open(file_name, 'a')
	try:
		myfile.write(' %d %f %d %s\n' % ( Gi_type,Gi_shift,tfunc,'- Gi version, reference Gi and type of logistic fucntion.'))
		myfile.write('%s\n' % (' 1 - number of chemical species in the system.'))
		myfile.write(' %s %f\n' % (element,atomic_weight))
		myfile.write('%s %s %s %s \n' % (" 0 0.50000",str(Rc),str(Tc),str(sigma)))  #NOTE THIS IS HARD CODED AND NEEDS TO BE FIXED
		myfile.write(' %s'%  (len(ro)))
		for i in range(0,len(ro)):
			myfile.write(' %6.4f'%  (ro[i]))
		myfile.write('%s\n' % (''))
		myfile.write(' %s '%  (len(NNlayers)))
		for i in range(0,len(NNlayers)):
			myfile.write('%s '%  (NNlayers[i]))
		myfile.write('%s\n' % (''))	
		for i in range(0,len(W)):
			myfile.write('%16.8e %8.4f\n'%  (W[i],0.0))		
	finally:
   		myfile.close()
	return 


