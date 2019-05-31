# THIS FILE IS LOADED (AS NEEDED) BY SUBROUTINES TO GIVE THEM ACCESS 
# TO GLOBAL PARAM DEFINED OR READ HERE
# THIS FILE SHOULDNT IMPORT ANY MODULES THAT IMPORT IT 

import numpy as np
from pylab import show
import matplotlib.pyplot as plt
from  os import  path, makedirs
from shutil import rmtree 
from random import shuffle

import torch
from torch.autograd import Variable

from input_file import *  
from write import *    
from read import *   #THESE SHOULDNT IMPORT initialize.py (circular)
from functions import *


#ONLY WANT TO RUN ONCE AT THE BEGINNING (I.E DONT RERUN IF VARIABLES ARE ALREADY DEFINED)
try: 
	print(dtype,"ALREADY DEFINED") #this will cause an error (if dtype not defined) and then python will try the except below
except: 

#------------------------------------------------------------------------
#MISC
#------------------------------------------------------------------------
	#####MAKE OUTPUT LOCATION (outdir defined in global_var.py)
	if path.exists(outdir):
		rmtree(outdir)
	makedirs(outdir)

	#CHECK FOR CUDA 
	if torch.cuda.is_available():
		write_to_log("# CUDA AVAILABLE");
		torch.cuda.empty_cache()
		dtype = torch.cuda.FloatTensor
		#dtype = torch.FloatTensor
	else:
		dtype = torch.FloatTensor

	if(pot_type!='NN'): # this code only does PINN,NN and BOP 
		write_to_log("ERROR: REQUESTED MODEL NOT CODED (EXITING)"); exit()

	write_to_log("# NUM THREADS="+str(torch.get_num_threads()))
	
#------------------------------------------------------------------------
#READ NN INFO 
#------------------------------------------------------------------------

	#READ NN FILE
	write_to_log("# READING NN FILE FROM FILE: "+str(NN_file))
	NN_INFO=read_NN_file(NN_file);  #read NN file and if NN is not defined there then randomize it 
	[NNlayers,param,ro,W]=NN_INFO   #UNPACK
	[Gi_type,Gi_shift,tfunc,element,atomic_weight,Rc,Tc,sigma,Nro]=param  #UNPACK MORE
	if(irandomize==1):
		W=1.0*np.random.uniform(-wb_bnd,wb_bnd,len(W)); 
	Wb=matrix_extract(W,NNlayers)

	#CONSISTENCY CHECKTS
	if( pot_type=='NN' and NNlayers[len(NNlayers)-1]!=1): write_to_log("ERROR: NN OUTPUT DIMENSION INCORRECT"); exit();
	
	#WRITE READ DATA TO LOG FILE
	write_input([pot_type,wb_bnd,e_shift,NNlayers,param,ro,W,Gi_type,Gi_shift,tfunc,element,atomic_weight,Rc,Tc,sigma,Nro])	


	#DEFINE SUB MARTICIES AND SET GRADIENTS
	Req_grad=True
	if(len(NNlayers)==3):
		w1=Variable(torch.tensor(Wb[0]).type(dtype), requires_grad=Req_grad);
		b1=Variable(torch.tensor(Wb[1]).type(dtype), requires_grad=Req_grad);
		w2=Variable(torch.tensor(Wb[2]).type(dtype), requires_grad=Req_grad);
		b2=Variable(torch.tensor(Wb[3]).type(dtype), requires_grad=Req_grad);
		write_to_log("# 	SUBMATRIX-SIZES: "+str(w1.shape)+' '+str(b1.shape)+' '+str(w2.shape)+' '+str(b2.shape))
		model_param=[w1,b1,w2,b2]
	elif(len(NNlayers)==4): 	
		#print(Wb[0])
		w1=Variable(torch.tensor(Wb[0]).type(dtype), requires_grad=Req_grad);
		b1=Variable(torch.tensor(Wb[1]).type(dtype), requires_grad=Req_grad);
		w2=Variable(torch.tensor(Wb[2]).type(dtype), requires_grad=Req_grad);
		b2=Variable(torch.tensor(Wb[3]).type(dtype), requires_grad=Req_grad);
		w3=Variable(torch.tensor(Wb[4]).type(dtype), requires_grad=Req_grad);
		b3=Variable(torch.tensor(Wb[5]).type(dtype), requires_grad=Req_grad);
		model_param=[w1,b1,w2,b2,w3,b3]
		write_to_log("# 	SUBMATRIX-SIZES: "+str(w1.shape)+' '+str(b1.shape)+' '+str(w2.shape)+' ' \
		+str(b2.shape)+' '+str(w3.shape)+' '+str(b3.shape))
	else:
		write_to_log("EXITING CODE: CURRENTLY HARDCODE ONLY FOR EITHER SINGLE OR DOUBLE HIDDEN LAYER NN"); exit()

#------------------------------------------------------------------------
#READ LS PARAM
#------------------------------------------------------------------------
	#READ GANGA"S NEIBH LIST FILE FOR COMPARISON 
	#structs and Gi are all dictionaries all with same keys SID=0,1 ... Nstr-1
	#Gi[0]-->list of lists w/ Gi for each atom in structure 0
	#structs[0]--> list=[N,E_DFT_(shifted),group,v,group_ID]

	#READ TRAINING SET 
	write_to_log("# READING NB LIST FROM FILE: "+str(LSPARAM_file))
	[Nat,Gi,structs]=read_LSPARAM(LSPARAM_file,NN_INFO) #NN_INFO INCLUDED FOR CONSISTENCY CHECK

	Nstr=len(structs); 
	write_to_log("# 	FILE CONTAINS: N_STRUCTURES="+str(Nstr)+'    N_ATOMS='+str(Nat))
	write_to_log("# 	BREAKING TOTAL SET INTO "+str(percent_train*100)+'% TRAINING AND '+str(np.ceil((1.0-percent_train)*100.0))+"% VALIDATION")

	#EXTRACT TRAINING SET
	train_indices=np.random.choice(len(structs),int(percent_train*len(structs)), replace=False) #  set
	train_indices = list(range(len(structs)))
	
	#VALIDATION SET  (structures not included in training)
	val_indices=[];
	for i in structs.keys(): 
		if(i not in train_indices): val_indices.append(i)
	write_to_log("# 		N_TRAIN_STRCT="+str(len(train_indices))+' N_TEST_STRCT='+str(len(val_indices)))

#-------------------------------------------------------------------------------
#BUILD ARRAYS FOR MODEL 
#-------------------------------------------------------------------------------
	data={}   #dictionary for relevant arrays/data 
	Nat=0.0
	Ns=len(train_indices)
	for SID in train_indices: #loop over structures IDs (SID)
		Nat=Nat+structs[SID][0] 

	R=torch.zeros(len(train_indices),int(Nat)).type(dtype); #REDUCTION TENSOR (THERE IS PROBABLY BETTER WAY TO DO THIS)

	y=[]; k=0; Ninv=[]; V_N=[];  i=0; grp_weighs=[]; x=[];
	for SID in train_indices: #loop over structures
		y.append(structs[SID][1]) #vector of structure energies
		V_N.append(structs[SID][3])  #volumes of each structure
		Ninv.append(1.0/structs[SID][0]) #vector of 1/N for each structure
		if(structs[SID][2] in weights.keys()):
			grp_weighs.append(weights[structs[SID][2]])	
		else: 
			grp_weighs.append(1.0)	
		for j in Gi[SID]:
			x.append(j) #vector of structure's Gi' energies 
			R[i][k]=1
			k=k+1
		i=i+1

	R=R.type(dtype);  

	y=torch.tensor(np.transpose([y])).type(dtype) 
	Ninv=torch.tensor(np.transpose([Ninv])).type(dtype)
	grp_w=torch.tensor(np.transpose([grp_weighs])).type(dtype) 
	x=torch.tensor(x).type(dtype);  
	M1=torch.tensor([np.ones(len(x))]).type(dtype)
	v=V_N; 


