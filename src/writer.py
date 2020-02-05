from  	os 			import  	path,getcwd 
from	datetime	import		datetime
from 	string		import 		Template
from 	json		import		dump
#from	pyfit 		import		*

#MISC PARAM
run_path	=	getcwd()+'/'											#tell module where to write
start_time 	=	str(datetime.now().strftime("%Y-%m-%d-H%H-M%M-S%S")) 	#file name prefix
prefix		=	'00' 													#start_time

def write_header():
	log("--------------------------------------------------------------")
	log("-------------------------PYFIT-FF-----------------------------")
	log("--------------------------------------------------------------")



def write_E_vs_V(data,t):
	with open(run_path+prefix+"-e_vs_V-"+str(data.name)+"-"+str(t)+".dat", 'w') as f:
		for i in range(0,len(data.v1)): #loop over structures
			f.write('%10f %10f %10f %5d %40s \n' % (data.v1[i],data.u1[i],data.u2[i],data.SIDS1[i],data.GIDS1[i]))	

def write_stats(name,t,RMSE,MAE,MED_AE,STD_AE,MAX_AE,RMS_DU):
	with open(run_path+prefix+"-stats-"+str(name)+".dat", 'a') as f:
		f.write('%10f %10f %10f %10f %10f %10f %10f \n' % (t,RMSE,MAE,MED_AE,STD_AE,MAX_AE,RMS_DU))	


def log(x,tab=0,name="-log.dat"):
	str_out=''
	if(str(type(x))=="<class 'list'>"):
		#res += '%50s\n'%str(x[i])
		for i in range(0,len(x)): str_out=str_out+'%s	'%str(x[i])
	else:
		str_out=str(x)
	with open(run_path+prefix+name, 'a') as f:
		if(tab==1):	
			f.write('\t  %s \n' % (str_out)); print("\t",str_out);
		else:		
			f.write('%s \n' % (str_out));    print(str_out)

def log_err(x): 
	str1=''
	for i in range(0,len(x)):
		str1=str1+' %14.10s'%str(x[i])
	log(str1,0,"-err-log.dat")

def log_dict(x):	#x=dictionary
	for i in x.keys():	log(['%-20s	:'%i ,x[i]],1)

def write_group_summary(x):	#x=dictionary
	with open(run_path+prefix+"-data-summary.dat", 'a') as f:
		f.write('%s \n' % ("# GROUP_ID : ATOM/STRUCTURE  N_STRUCTURES  N_ATOMS_IN_GROUP"));  #  print(str_out)
		for i in x.keys():	f.write('%s	%d	%d	%d \n' % ('%35s :'%i,x[i][0],x[i][1],x[i][2]))

def write_LSP(str1):
	with open(run_path+prefix+"-LSP.dat", 'a') as f:
		f.write('%s \n' % (str1)); #print("\t",str_out);

def write_NN(nn,step):
	WB=nn.matrix_combine()
	with open(run_path+prefix+"-NN-"+str(step)+".dat", 'w') as f:
		f.write(' %d %f %d\n' % 		(nn.info['lsp_type']	,		nn.info['lsp_shift'],	 	nn.info['activation']))
		f.write(' %d \n' % 				(nn.info['num_species']	))
		f.write(' %s %f\n' % 			(nn.info['species']	 	,		nn.info['atomic_weight']))
		f.write(' %s %f %f %f %f \n' % ("0",nn.info['max_rand_wb'],	nn.info['cutoff_dist'],		nn.info['cutoff_range'],	nn.info['lsp_sigma'])) 
		f.write(' %s'	%  (len(nn.info['lsp_lg_poly'])))
		for i in nn.info['lsp_lg_poly']:	f.write(' %d'%  (i))
		f.write('\n %s'	%  (len(nn.info['lsp_ro_val'])))
		for i in nn.info['lsp_ro_val']:	f.write(' %6.5f'%  (i))
		f.write('\n %s'	%  (int(nn.info['ibaseline'])))
		for i in nn.info['bop_param']:	f.write(' %8.6f'%  (i))
		f.write('\n %d'	%  (len(nn.info['nn_layers'])))
		for i in nn.info['nn_layers']:	f.write(' %d'%  (i))
		f.write('\n')
		for i in WB:	f.write('%16.8e %8.4f\n'%  (i,0.0))
		f.write('\n')


def write_poscar(x):
	# x=structure object
	if(str(type(x))!="<class 'dataset.Structure'>"): raise ValueError("EXPECTED STRUCTURE OBJECT BUT GOT SOMETHING ELSE")
	#WRITE POSCAR FOR GIVE STRUCTURE
	with open(run_path+str(x.comment)+'-'+str(x.sid)+'.POSCAR', 'w') as f:
		f.write('%s\n' % (x.comment))
		f.write('%s\n' % (x.scale_factor))
		f.write('%1.10f %1.10f %1.10f \n' % (x.a1[0],x.a1[1],x.a1[2])) #shift cell and atoms to zero
		f.write('%1.10f %1.10f %1.10f \n' % (x.a2[0],x.a2[1],x.a2[2]))  
		f.write('%1.10f %1.10f %1.10f \n' % (x.a3[0],x.a3[1],x.a3[2]))  
		f.write('%s \n' % (x.species))  
		f.write('%d \n' % (x.N))  
		string='cartesian' #if(x.is_cartesian) else 'direct'
		f.write('%s\n' % (string))
		for ri in x.positions:
			f.write('%1.10f %1.10f %1.10f \n' % (ri[0],ri[1],ri[2]))  
	# #WRITE FORCES TO IF AVAILABLE
	# if(x.forces_avail):
	# 	with open(run_path+str(x.comment)+'-'+str(x.sid)+'.FORCES', 'w') as f:
	# 		for ri in x.forces:
	# 			f.write('%1.10f %1.10f %1.10f \n' % (ri[0],ri[1],ri[2]))  
