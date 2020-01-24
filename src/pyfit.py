#!/usr/bin/env python3
#Authors: James Hickman (NIST) and Adam Robinson (GMU)


from	sys			import		argv
import 	torch
import  numpy		as			np

import  reader
import  writer
import	util


#-------------------------------------------------------------------------
#PART-1: SETUP
#-------------------------------------------------------------------------

writer.write_header()

#CHECK THAT A FILE WAS PROVIDED BY USER
if(len(argv)!=2):	raise ValueError("no input file or too many command line arguments provided")

#SB=SNOWBALL DICTIONARY THAT ACCUMULATED EVERYTHING
SB={};		SB['input_file']=argv[1]

##GET RUN PARAMETER AND DEFAULTS (see src/util.py)
util.get_run_parameters(SB)			 
util.get_defaults(SB)			 


#READ INPUT FILES
reader.read_input(SB)			#READ INPUT FILE AND ADD INFO TO SB
reader.read_nn_file(SB)			#READ NN FILE AND ADD INFO TO SB 
reader.read_database(SB);		#READ DATABASES AND ADD INFO TO SB 

#COMPUTE NEIGHBORLIST (NBL) AND LSP FOR ALL STRUCTURES
util.compute_all_nbls(SB)			
util.compute_all_lsps(SB)	
 

util.partition_data(SB)
util.construct_matrices(SB)

# print(SB['test_SIDS'])
exit()




#if(SB['train_edges']): 	util.find_outliers_1(SB)	 
if(SB['train_edges']): 	util.find_outliers_1(SB)	 

exit()

#FORM TRAINING AND TEST SET 
util.split_data_train_validation(SB)

exit()
# #COMPUTE LSP

#FORM DATA


print(SB['nn'])
exit()

# SB['nn_file_path']);	SB.update(nn.info);		#READ NN FILE 


# #READ DATASET FILE
# [dataset_info,group_sids,group_info,structures]=reader.read_database(dataset_path);		SIDS=structures.keys()				
# locals().update(dataset_info)



exit()
#-------------------------------------------------------------------------
#SETUP-PART-2: DO VARIOUS INITIAL CONSISTENCY CHECKS
#-------------------------------------------------------------------------
writer.log("CHECKING FOR ERRORS:");  										
if(pot_type not in ['NN']): 				raise ValueError("POT_TYPE="+str(pot_type)+" NOT CODED") 
if(pot_type=='NN' and nn_layers[-1]!=1): 	raise ValueError("NN OUTPUT DIMENSION INCORRECT");

writer.log("	NO ERRORS FOUND:");  										


#-------------------------------------------------------------------------
#SETUP-PART-3: COMPUTE NEIGHBOR LIST AND LOCAL STRUCTURE PARAMETERS
#-------------------------------------------------------------------------

#SPLIT DATA SET INTO TRAINING AND VALIDATION
[training_structures,validation_structures]=util.split_data_training_validation(SIDS,group_sids)

training_set=DataSet(training_structures)




#-------------------------------------------------------------------------
#SETUP-PART-2: CREATE ADDITIONAL STRUCTURES USED FOR FORCE TRAINING
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
#SETUP-PART-4: 
#-------------------------------------------------------------------------

exit()

#-------------------------------------------------------------------------
#SETUP-PART-2: VARIOUS ADDITIONAL CONSISTENCY CHECKS
#-------------------------------------------------------------------------
writer.log("CHECKING FOR MORE ERRORS:");  										 
if(len(training_structures)+len(validation_structures)!=len(structures.keys())): 	raise ValueError("LOST STRUCTURE") 
writer.log("	NO ERRORS FOUND:");  										

#-------------------------------------------------------------------------
#SETUP-PART-2: BUILD ARRAYS NEEDED FOR TRAINING
#-------------------------------------------------------------------------

# def build_array(indices):



# for i in 


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------TRAINING-----------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

exit()
print(n_iter_save)

#DUMP ALL READ PARAMTERS INTO THIS DICTIONARY (MAKES WRITING LOG CLEANER)
param={}











exit()


# param=initializer.setup(argv); 		locals().update(param)

# print(nn['info'])
# print(nn['values'])
# poscar_data = PoscarLoader(e_shift)
# poscar_data = poscar_data.loadFromFile(dataset_path)

# print(poscar_data.structures[0].n_atoms)



exit()









# from	time		import 	time,sleep
# import	sys 	
# import	numpy		as		np
# import	torch

# import	initialize  


# initialize.setup()
# # #The initialize.setup() (see initialize.py) function gets everything ready for training  
# # # 	-i.e. reads files, creates output folder, does sanity checks, and forms data structures 
# # #	-the setup function retuns all variables that it creates (i.e locals())
# # locals().update(initialize.setup())		#save variables made in setup()to main.py locals

# print("here")
# # print(nn_file_path)

# exit()

# global INPUT_PATH

# INPUT_PATH	=sys.argv[1]







# ###IMPORT FUNCTIONS FROM SUBROUTINES
# path.append('subroutines') 

# from input_file import *   #READ INFORMATION FROM (see input_file.py)
# from write import *    

# #THE FOLLOWING IMPORT STATEMENT IS VERY IMPORTANT, IT RUNS THE initialize.py SCRIPT WHICH
# #DEFINES (OR READS) ALL GLOBAL VARIABLES AND GIVES VARIOUS SCRIPTS ACCESS TO THEM (see initialize.py)
# from initialize import *  
# from functions import *

exit()



















#------------------------------------------------------------------------
#DEFINE MODEL
#------------------------------------------------------------------------
def model():
	ypred=0; del ypred

	#MATHEMATICAL NN 
	if(pot_type=='NN'):
		#EVALUATE NN
		if(tfunc==1 and len(NNlayers)==3):
			ypred=R.mm((torch.sigmoid(x.mm(torch.t(w1))+torch.t(b1.mm(M1)))-0.5).mm(torch.t(w2))+torch.t(b2.mm(M1)))
		elif(tfunc==0 and len(NNlayers)==3):
			ypred=R.mm((torch.sigmoid(x.mm(torch.t(w1))+torch.t(b1.mm(M1)))).mm(torch.t(w2))+torch.t(b2.mm(M1))) 
		elif(tfunc==1 and len(NNlayers)==4):
			ypred=R.mm((torch.sigmoid((torch.sigmoid(x.mm(torch.t(w1))+torch.t(b1.mm(M1)))-0.5).mm(torch.t(w2)) 
			+torch.t(b2.mm(M1)))-0.5).mm(torch.t(w3))+torch.t(b3.mm(M1)))
		elif(tfunc==0 and len(NNlayers)==4):
			ypred=R.mm((torch.sigmoid((torch.sigmoid(x.mm(torch.t(w1))+torch.t(b1.mm(M1)))).mm(torch.t(w2)) 
			+torch.t(b2.mm(M1)))).mm(torch.t(w3))+torch.t(b3.mm(M1)))
		else:
			print("EXITING CODE: MNN SIZE NOT YET CODED"); exit() 

	return ypred


#------------------------------------------------------------------------
#CLOSURE FUNCTION
#------------------------------------------------------------------------
def closure():
	global RMSE,y1

	if(opt_alg=='LBFGS'):
		optimizer.zero_grad()

	loss=0.0; RMSE=0.0; y1=0.0
	S=0.0; N=0.0; #S1=0.0; N1=0.0; S2=0.0; N2=0.0

	y1=model() #.cpu()
	RMSE=torch.sqrt((grp_w*(torch.mul((y1-y),Ninv)**2.0)).sum()/Ns); 
	print(t,RMSE.item(),batch_i,switch,"TIME") #,time())

	if(str(RMSE.item())=='nan'):
		print("RMSE=NAN (EXITING):"); exit()

	loss=RMSE

	if(opt_alg=='LBFGS'): # and switch!=1):
		loss.backward()

	RMSE=RMSE.detach()
	return loss

#------------------------------------------------------------------------
#OPTIMIZATION 
#------------------------------------------------------------------------

#PRINT HEADER LINE
with open(logfile, 'a') as out:
	out.write('%s\n' % ('#iteration  RMSE(eV/atom)  time(s)'))

RMSE=1000; t=0;  batch_i=Nbatch-1; #send(batch_i)  #SEND ARRAYS TO GPU
max_iter=100

#OPTIMIZATION CHOICE
alpha=0.01 #LEARNING RATE
opt_alg='LBFGS' #FITTING ALG
#opt_alg='MISC'

write_to_log("ALG="+opt_alg+" LR="+str(alpha))

if(pot_type=='NN'):
	if(len(NNlayers)==3):
		if(opt_alg=='LBFGS'):
			optimizer=optim.LBFGS([w1,w2,b1,b2], lr=alpha) 
		else: 
			optimizer = optim.SGD([w1,w2,b1,b2], lr = 0.001, momentum=0.9)
			#optimizer = optim.Adam([w1,w2,b1,b2], lr = alpha)
	elif(len(NNlayers)==4): 
		if(opt_alg=='LBFGS'):
			optimizer=optim.LBFGS([w1,w2,w3,b1,b2,b3], lr=alpha, max_iter=10) 
		else: 
			optimizer = optim.SGD([w1,w2,w3,b1,b2,b3], lr =0.001, momentum=0.9) #alpha=0.1;  #LEARNING RATE
			#optimizer=optim.Adam([w1,w2,w3,b1,b2,b3], lr=0.0001) 

switch=1
start=time()
while(t<max_iter): # and RMSE>0.003):

	#--------------------------------------------------------------
	# STEP
	#--------------------------------------------------------------
	if(opt_alg=='LBFGS'):
		optimizer.step(closure)
	else: 
		optimizer.zero_grad()
		loss = closure()
		loss.backward()
		optimizer.step()

#	##--------------------------------------------------------------
#	##WRITE OUTPUT 
#	##--------------------------------------------------------------
	isave1=1
	if(t%isave1==0): #ALSO PRINT FINAL ITERATION

		with open(errlogfile, 'a') as out:
			out.write('%4d %20f %10f \n' % (t,RMSE,time()-start))

		isave2=isave1*20
		if(t%isave2==0):
			with open('output/E_vs_V-CURRENT.dat', 'w') as out:
				for i3 in range(0,len(v)):
					out.write('%10f %10f %10f \n' % (v[i3],y[i3].item()*Ninv[i3].item(),y1[i3].item()*Ninv[i3].item()))

			with open('output/E_vs_V-'+str(t)+'.dat', 'a') as out:
				for i3 in range(0,len(v)):
					out.write('%10f %10f %10f \n' % (v[i3],y[i3].item()*Ninv[i3].item(),y1[i3].item()*Ninv[i3].item()))

	t=t+1

print('TIME:',time()-start)



# ri=torch.tensor([0.0,0.0,0.0], requires_grad = True)
# nbl=torch.tensor([[1.0,2.1,4.0],[3.0,2.1,4.0]])
# nbl=nbl-ri



# exit()



# gi = torch.tensor([[1.0,2.1,4.0],[3.0,2.1,4.0]], requires_grad = True)
# #e = torch.sum(torch.sin(gi**3.0),1).view(2,1)
# e = torch.sum(torch.sin(gi**3.0),1).view(2,1)

# print(gi.shape,e.shape )
# e.backward(gi)
# print(gi.grad/gi.detach()) 

# exit()

# print(torch.cos(x**3.0)*3.0*x**2.0)
# exit()


# A=torch.tensor(np.array([3.0,11.0,20.0])).view(3,1) 
# B=A.view(1,3)
# print(((A-B)**2.0)**0.5)
# print(A/A)
# print(A-B)
# print(A,B)
# exit()
