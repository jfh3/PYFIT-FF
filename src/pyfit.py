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
rmse		= 	1000.0
SB['nn'].set_grad()

#WRITE INITIAL DATA
util.chkpnt(SB,t); t=t+1  
if(max_iter==0): exit()	 #DONT START LOOP

# exit()

# def set_optim()"

#HARDED-CODED TO USE LBFGS (SEEMS TO BE THE BEST) 
if(SB['pot_type']=='NN'):
	#SMOOTHLY INCREASE LR (MORE STABLE FITTING)
	if(SB['ramp_LR']):
		optimizer=optim.LBFGS(SB['nn'].submatrices, lr=1.0) 
		mid_ramp=25; LR_i=0.0001
		lmbda = lambda t: SB['LR']*(np.tanh(6.0*(t-mid_ramp)/mid_ramp)+1.0)/(2.0+2.0*LR_i)+LR_i 
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda,-1)
	else: 
		optimizer=optim.LBFGS(SB['nn'].submatrices, lr=SB['LR_i']) 

def closure():
	global loss,OBE1,OBL1,OBL2,OBL3,OB_DU,rmse,OBT
	optimizer.zero_grad(); loss=0.0 
	[rmse,OBE1,OB_DU,OBL1,OBL2,OBL3]=training_set.compute_objective(SB)
	loss=OBE1+OB_DU+OBL1+OBL2+OBL3

	loss.backward();
	OBE1=OBE1.item();	OB_DU=OB_DU.item();	OBL3=OBL3.item()
	OBL1=OBL1.item();	OBL2=OBL2.item();	OBT=loss.item();

	if(str(OBE1)=='nan' or rmse>10**10 ):
		writer.log(['%10.7s'%str(t),'%10.7s'%str(rmse),'%10.7s'%str(OBE1), \
			    '%10.7s'%str(OBL1),'%10.7s'%str(OBL2)],0,"-err-log.dat")
		raise ValueError("OB1=NAN or OB1>10000000000 (TRY LOWER LR)")

	return loss

#OPTIMIZATION LOOP
start=time(); #RMSE=1
writer.log('STARTING FITTING LOOP:')
writer.log(["	INITIAL LR:",'%10.7s'%str(optimizer.param_groups[0]['lr'])])

while(t<max_iter):  # and RMSE>RMSE_FINAL and ISTOP==False): 

	optimizer.step(closure)

	if(SB['ramp_LR']): scheduler.step()

	writer.log(['%10.7s'%str(t),'%10.7s'%str(rmse),'%10.7s'%str(OBE1),'%10.7s'%str(OB_DU), \
		    '%10.7s'%str(OBL1),'%10.7s'%str(OBL2),'%10.7s'%str(OBL3),'%10.7s'%str(OBT), \
		    '%10.7s'%str(optimizer.param_groups[0]['lr'])],0,"-err-log.dat")

	if(t%SB['save_every']==0):  util.chkpnt(SB,t);   

	if(((rmse_last-rmse)**2.0)**0.5<SB['rmse_tol'] or rmse<SB['rmse_final']): 
		writer.log("STOPPING CRITERION MET:"); t=max_iter
	rmse_last=rmse

	t=t+1

writer.log(['FITTING LOOP TIME (SEC):',time()-start])
t=111111 #FOR CONSISTANT FILE NAMING
util.chkpnt(SB,t)

exit()


