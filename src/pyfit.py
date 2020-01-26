#!/usr/bin/env python3
#Authors: James Hickman (NIST) and Adam Robinson (GMU)

from 	time	import	time,sleep
from	sys	import	argv
import  numpy	as	np

import  reader
import  writer
import	util
import  torch.optim as optim

#-------------------------------------------------------------------------
#PART-1: SETUP
#-------------------------------------------------------------------------

writer.write_header()

#CHECK THAT A FILE WAS PROVIDED BY USER
if(len(argv)!=2):	raise ValueError("NO INPUT FILE")

#SB=SNOWBALL (i.e DICTIONARY THAT ACCUMULATED EVERYTHING)(SB != sackville baggins) 
SB={};		
SB['input_file']=argv[1]

##GET RUN PARAMETER 
util.get_run_parameters(SB)	
#READ INPUT FILES
reader.read_input(SB)		#READ INPUT FILE AND ADD INFO TO SB
reader.read_pot_file(SB)	#READ NN FILE AND ADD INFO TO SB 
reader.read_database(SB);	#READ DATABASES AND ADD INFO TO SB 

#WRITE POSCAR IF DESIRED 
if(SB['dump_poscars']):	util.dump_poscars(SB)()

#COMPUTE NEIGHBORLIST (NBL) AND LSP FOR ALL STRUCTURES
util.compute_all_nbls(SB)	
util.compute_all_lsps(SB)	
util.partition_data(SB)


#-------------------------------------------------------------------------
#PART-1: Train
#-------------------------------------------------------------------------

t		=	0;  
max_iter	=	SB['max_iter']
training_set	=	SB['training_set']
rmse_last	=	100
rmse		= 	1000
SB['nn'].set_grad()

#WRITE INITIAL DATA
util.chkpnt(SB,t); #t=t+1  
if(max_iter==0): exit()	 #DONT START LOOP

#HARDED-CODED TO USE LBFGS (SEEMS TO BE THE BEST) 
if(SB['pot_type']=='NN'):
	optimizer=optim.LBFGS(SB['nn'].submatrices, lr=SB['learning_rate']) 

def closure():
	global loss,OBE1,OBL1,OBL2,OB_DU,rmse
	optimizer.zero_grad(); loss=0.0 
	[rmse,OBE1,OB_DU,OBL1,OBL2]=training_set.compute_objective(SB)
	loss=OBE1+OB_DU+OBL1+OBL2
	loss.backward();
	OBE1=OBE1.item();	OB_DU=OB_DU.item()
	OBL1=OBL1.item();	OBL2=OBL2.item()

	if(str(OBE1)=='nan' or rmse>10**10 ):
		writer.log(['%10.7s'%str(t),'%10.7s'%str(rmse),'%10.7s'%str(OBE1), \
			    '%10.7s'%str(OBL1),'%10.7s'%str(OBL2)],0,"-err-log.dat")
		writer.log("OB1=NAN or OB1>10000000000 (EXITING):"); exit()

	return loss

#OPTIMIZATION LOOP
start=time(); #RMSE=1
writer.log('STARTING FITTING LOOP:')

while(t<max_iter): #and RMSE>0.132299): # and RMSE>RMSE_FINAL and ISTOP==False): 



	#if(t<100):
	#	SB['xtanhx']=False
	#else:
	#	SB['xtanhx']=True
	optimizer.step(closure)

	writer.log(['%10.7s'%str(t),'%10.7s'%str(rmse),'%10.7s'%str(OBE1),'%10.7s'%str(OB_DU), \
		    '%10.7s'%str(OBL1),'%10.7s'%str(OBL2)],0,"-err-log.dat")

	if(t%SB['save_every']==0):  util.chkpnt(SB,t);   

	if(((rmse_last-rmse)**2.0)**0.5<SB['rmse_tol']): 
		writer.log("CONVERGENCE CRITERION MET:"); t=max_iter
	rmse_last=rmse

	t=t+1

writer.log(['FITTING LOOP TIME (SEC):',time()-start])
t=111111 #FOR CONSISTANT FILE NAMING
util.chkpnt(SB,t)

exit()


