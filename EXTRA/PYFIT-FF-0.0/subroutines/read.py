#import as little code as needed
from numpy import array, random
from  os import  path 
from input_file import *  
from time import time
import numpy as np
from write import *    
#import torch
#from torch.autograd import Variable
##----------------READ NN FROM FILE-----------------
#READS NN FILES IN GANGA'S CURRENT FORMAT
# 1 0.5 1 - Gi method, reference Gi, and logistic function type
# 1 - number of chemical species in the system ( .le. 4)
# 'Si'  28.0855 - element symbol and weight
# 1 0.500000 5.0 0.5 1.000000
# 12 1.5000 1.7727 2.0455 2.3182 2.5909 2.8636 3.1364 3.4091 3.6818 3.9545 4.2273 4.5000
# 4 60 16 16 8

def read_NN_file(file_path): 
	if path.exists(file_path):
		file3=file_path
		input_file = open( file3, "r" )
		W=[]; NNlayers=[]; ro=[]; line_num=1
		for line in input_file:
			if(line_num==1):
				parts=line.split()
				Gi_type=int(parts[0])     # 0=(ORIGINAL Gi)  1=log(Gi+Gi_shift)
				Gi_shift=float(parts[1])  # (see above)
				tfunc=int(parts[2])       # 0=sigmoid 1=sigmod-0.5
			if(line_num==2):
				parts=line.split()
				Nspecies=int(parts[0])   
			if(line_num==3): #
				parts=line.split()
				element=str(parts[0])   
				atomic_weight=float(parts[1])   
			if(line_num==4):
				parts=line.split()
				irandomize=int(parts[0]) #DECIDE WHETHER TO READ OR BUILD THE NN (1=build)(0=read)
				#SECOND NUMBER IS IGNORED BY THIS VERSION OF  PYTHON
				Rc=float(parts[2])
				Tc=float(parts[3])
				sigma=float(parts[4])
			if(line_num==5):
				parts=line.split()
				for i in range(1,len(parts)):
					ro.append(float(parts[i]))
			if(line_num==6):
				parts=line.split()
				for i in range(1,len(parts)):
					NNlayers.append(int(parts[i]))
				if(int(parts[0]) != len(NNlayers)): write_to_log("ERROR IN NN-FILE (INCORRECT NUMBER OF LAYERS)");	exit()
				if(irandomize==1): 
					write_to_log("# 	NOTE: NO NN IN FILE THEREFORE RANDOMIZING NEURAL NETWORK")

					#DETERMINE N FITTING PARAM
					S=0
					for i in range(1,len(NNlayers)-1):
						S=S+NNlayers[i-1]*NNlayers[i]+NNlayers[i]
						#print("#Nodes in hidden layer-",i,":", NNlayers[i])
					S=S+NNlayers[len(NNlayers)-2]*NNlayers[len(NNlayers)-1]+NNlayers[len(NNlayers)-1]
					W= random.uniform(-wb_bnd,wb_bnd,S)



			if(line_num>6):
				if(irandomize==0): 
					#write_to_log("# 	READING NEURAL NETWORK FROM FILE")
					parts=line.split()
					if(parts!=[]):
						W.append(float(parts[0]))
			line_num=line_num+1
		Nro=len(ro)
		param=[Gi_type,Gi_shift,tfunc,element,atomic_weight,Rc,Tc,sigma,Nro]
		if(irandomize==0): W=array(W)
	else: 
		write_to_log("ERROR: COULDNT FIND NN FILE: "+str(file_path)+" (EXITING CODE)");	exit()

	if(irandomize==1): 
		#WRITE NN TO FILE (SAVE RANDOMIZATION)
		write_NN(W,NNlayers,ro,param,outdir+'/nn0.dat')

	return [NNlayers,param,ro,W]


def read_LSPARAM(file_path,NN_INFO): 
	start=time()
	[NNlayers,param,ro,W]=NN_INFO #UNPACK
	[Gi_type,Gi_shift,tfunc,element,atomic_weight,Rc,Tc,sigma,Nro]=param  #UNPACK MORE

	#NOTE: LOG(Gi+0.5) already included in LSPARAM file
	input_file = open( file_path, "r" )
	ro1=[];  str_IDs=[]; # atom_IDs=[]

	line_next=0;  Natoms=0; line_num=1;  
#	atoms={} #dictionary of dictionaries for atoms (given atom-id return atom info)
	structs={}; Gis={}; NBLs={};
	for line in input_file: #each line is an atom
		parts1=line.split()
		parts=parts1
		#READ HEADER 
		if(line_num==1):
			if(Gi_type!=int(parts[1])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (Gi_type) (EXITING CODE)");	exit()
			if(Gi_shift!=float(parts[2])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (Gi_shift) (EXITING CODE)"); exit()		
			if(tfunc!=int(parts[3])): write_to_log("ERROR: NN+NBL FILES DONT MATCH:(tfunc) (EXITING CODE)"); exit()			    	     
		if(line_num==3):
			if(element != str(parts[1])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (element) (EXITING CODE)"); exit()
			if(atomic_weight != float(parts[2])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (atomic_weight) (EXITING CODE)"); exit()  
		if(line_num==4):
			if(Rc != float(parts[3])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (Rc) (EXITING CODE)"); exit()  
			if(Tc != float(parts[4])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (Tc) (EXITING CODE)"); exit()  
			if(sigma != float(parts[5])): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (sigma) (EXITING CODE)"); exit()  
		if(line_num==5):
			for i in range(2,len(parts)):
				ro1.append(float(parts[i]))
			if(ro != ro1): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (ro) (EXITING CODE)"); exit()  
		if(line_num==6):
			
			#NB LIST IS ONLY A PROBLEM FOR PINN AND BOP (IN THIS CASE NEIRBOR LIST TYPE MUST MATCH)
			if(pot_type==2 or pot_type==0):
				if(int(parts[1])!=pot_type): write_to_log("ERROR: NN+NBL FILES DONT MATCH: (pot_type) (EXITING CODE)"); exit()  
		if(line_num==8):
			Natoms1=int(parts[1])

		if(line_num>9): 
			#GROUP-NAME GROUP-ID STRUCTURE-ID STRUCTURE-Natom STRUCTURE-E_DFT Gi

			if(parts[0][0:4]=="ATOM"):
				SID=int(parts[3])
				if(SID not in str_IDs):
#					print(int(parts[4]))
					Gi=[]; # NBL=[]
					str_IDs.append(int(parts[3]))
				 	#N E_DFT_(shifted) B01 v
					structs[int(parts[3])]=[float(parts[4]),float(parts[5])+e_shift*float(parts[4]),str(parts[1]),float(parts[6]),int(parts[2])]
				Natoms=Natoms+1
				line_next=line_num+1
			if(line_next==line_num and parts[0]=='Gi'):
				temp=[]
				for Gi1 in parts[1:len(parts)]: 
					temp.append(float(Gi1))
				Gi.append(temp)
				Gis[SID]=Gi  #will keep overwritting
				line_next=line_num+1

		line_num=line_num+1

	write_to_log("# 	TIME TO READ LSPARAM FILE(sec): "+str(time()-start))
	return [Natoms,Gis,structs]



