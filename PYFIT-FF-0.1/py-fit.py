from time import time,sleep
from sys import path
import numpy as np
import random
#IMPORT PYTORCH LIB
import torch
import torch.optim as optim

###IMPORT FUNCTIONS FROM SUBROUTINES
path.append('subroutines') 

from input_file import *   #READ INFORMATION FROM (see input_file.py)
from write import *    

#THE FOLLOWING IMPORT STATEMENT IS VERY IMPORTANT, IT RUNS THE initialize.py SCRIPT WHICH
#DEFINES (OR READS) ALL GLOBAL VARIABLES AND GIVES VARIOUS SCRIPTS ACCESS TO THEM (see initialize.py)
from initialize import *  
from functions import *
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


