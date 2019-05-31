
#------------------------------------------------------------------------
#--------------------------------USER INPUT------------------------------
#------------------------------------------------------------------------

##INPUT FILES 
path1='input/Rc4.5-Tc1.0-Gi2-EOS-NN';   pot_type='NN'    #4.146813

NN_file=str(path1)+'/nn1.dat'
LSPARAM_file=str(path1)+'/LSParam.dat' 

#OUTPUT PATHS
outdir='output'  #name of output directory
logfile=outdir+'/log.dat'
errlogfile=outdir+'/err_log.dat'
errlogfile2=outdir+'/err_log_full.dat'
min_max_file=outdir+'/weight-bias-min-and-max.dat'

#MISC
irandomize=0 #if 1 then overwrite the NN in file with random numbers
wb_bnd=0.1   #maximum weight and bias for NN initalization
percent_train=1.0  #uses this fraction of the data as a training set (the rest is validation)
e_shift=0.0

#CONTROL HOW THE DATA IS PARITIONED FOR BATCHS/BINS 
#(Nbatch=1 --> batch training scheme)
#(Nbatch>1 --> mini-batch training scheme)
Nbatch=1   #number of "mini-batches" (i.e. random subsets of training set) 
if(Nbatch==1):
	switch_batch_every=20000000 #never switch
if(Nbatch>1):
	switch_batch_every=10  

#SUBGROUP WEIGHTS 
weights={}
weights['Si_B1']=1.0
weights['Si_B15']=1.0
weights['Si_B00']=1.0
weights['Si_B01']=1.0
weights['Si_B02']=1.0

weights['Si_L1']=1.0
weights['Si_C1']=1.0

