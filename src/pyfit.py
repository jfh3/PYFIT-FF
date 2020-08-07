#!/usr/bin/env python3
#Authors: James Hickman (NIST) and Adam Robinson (GMU)

from 	time	import	time,sleep
from	sys	import	argv
import  numpy	as	np

import  reader
import  writer
import	util
import  torch.optim as optim
import 	torch 

#-------------------------------------------------------------------------
#PART-1: SETUP
#-------------------------------------------------------------------------
writer.write_header()

#CHECK THAT A FILE WAS PROVIDED BY USER
if(len(argv)!=2):	raise ValueError("NO INPUT FILE")

#SB=SNOWBALL (i.e DICTIONARY THAT ACCUMULATED EVERYTHING) 
SB={};		
SB['input_file']=argv[1]

##GET RUN PARAMETER 
util.get_run_parameters(SB)

#READ INPUT FILES
reader.read_input(SB)		#READ INPUT FILE AND ADD INFO TO SB
reader.read_pot_file(SB)	#READ NN FILE AND ADD INFO TO SB 
reader.read_database(SB);	#READ DATABASES AND ADD INFO TO SB 

#WRITE INDIVIDUAL POSCAR IF DESIRED 
if(SB['dump_poscars']):	util.dump_poscars(SB)()

#COMPUTE NEIGHBORLIST (NBL) AND LSP FOR ALL STRUCTURES
util.compute_all_nbls(SB)	
util.compute_all_lsps(SB)
util.partition_data(SB)

# #SAVE SB
writer.write_json(SB)
exit()

#-------------------------------------------------------------------------
#PART-2: TRAIN
#-------------------------------------------------------------------------

#INITIALIZE MISC PARAM
t		=	0;  
rmse		= 	1000.0


#PARAM FOR CHECKING RMSE_TOL
check_every	=	100
rmse_m1		=	0
rmse_m2		=	0
m1_turn		=	True


SB['nn'].set_grad()

#WRITE INITIAL DATA
util.chkpnt(SB,t); t=t+1  
if(SB['max_iter']==0): exit()	 #DONT START LOOP

#HARDED-CODED TO USE LBFGS (SEEMS TO BE THE BEST) 
def set_optim():
	global optimizer,scheduler
	if(SB['pot_type']=='NN'):
		#SMOOTHLY INCREASE LR (MORE STABLE FITTING)
		if(SB['ramp_LR']):
			optimizer=optim.LBFGS(SB['nn'].submatrices, max_iter=SB['lbfgs_max_iter'], lr=1.0) 
			mid_ramp=SB['mid_ramp']; LR_i=SB['LR_o']
			lmbda = lambda t: SB['LR_f']*(np.tanh(6.0*(t-mid_ramp)/mid_ramp)+1.0)/(2.0+2.0*LR_i)+LR_i 
			scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda,-1)
		else: 
			optimizer=optim.LBFGS(SB['nn'].submatrices, max_iter=SB['lbfgs_max_iter'], lr=SB['LR_f']) 
set_optim()

def closure():
	global loss,OBE1,OBL1,OBLP,OB_DU,rmse,OBT
	optimizer.zero_grad(); loss=0.0 
	[rmse,OBE1,OB_DU,OBL1,OBLP]=SB['training_set'].compute_objective(SB)
	loss=OBE1+OB_DU+OBL1+OBLP
	loss.backward();
	OBE1=OBE1.item();	OB_DU=OB_DU.item();	OBLP=OBLP.item()
	OBL1=OBL1.item();	OBT=loss.item();
	return loss

#OPTIMIZATION LOOP
start=time();  
writer.log('STARTING FITTING LOOP:')
writer.log(["	INITIAL LR:",'%10.7s'%str(optimizer.param_groups[0]['lr'])])
while(t<SB['max_iter']):  

	optimizer.step(closure)
	if(SB['ramp_LR']): scheduler.step() #ADJUST LR 

	#CHECK IF OBJ FUNC BLEW UP
	if(str(OBE1)=='nan' or rmse>1000000): #START OVER
		writer.log("NOTE: THE OBJ FUNCTION BLEW UP (STARTING OVER)(MAYBE TRY SMALLER LR)")
		SB['nn'].unset_grad();	SB['nn'].randomize();	set_optim();  

	#WRITE 
	if(rmse<100 and t%10==0): writer.log_err([t,rmse,OBE1,OB_DU,OBL1, \
				  OBLP,OBT,SB['nn'].maxwb,optimizer.param_groups[0]['lr']])  #
	if(t%SB['save_every1']==0 or t%SB['save_every2']==0):  util.chkpnt(SB,t);   

	#CONVERGENCE CRITERION-1 (REACHED RMSE THRESHOLD)
	if(rmse<SB['rmse_stop']): 
		writer.log("STOPPING CRITERION MET:"); t=SB['max_iter']

	#CONVERGENCE CRITERION-2 (CHANGE IN RMSE<THRESHOLD)
	delta1=((rmse_m1-rmse)**2.0)**0.5
	delta2=((rmse_m2-rmse)**2.0)**0.5
	if(delta1<SB['rmse_tol'] and delta2<SB['rmse_tol']): 
		writer.log("STOPPING CRITERION MET:"); t=SB['max_iter']
	if(t%check_every==0):	#UPDATE PREVIOUS RMSE CHCK_PNT 
		if(m1_turn):
			rmse_m1=rmse;	m1_turn=False; 
		else:
			rmse_m2=rmse;	m1_turn=True; 

	#UPDATE ITERATION COUNTER
	t=t+1

writer.log(['FITTING LOOP TIME (SEC):',time()-start])
t=111111 	#FOR CONSISTANT NAMING OF LAST FILE
util.chkpnt(SB,t)

exit()
