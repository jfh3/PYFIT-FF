import  writer
from 	torch	import	get_num_threads,cuda,FloatTensor
import	numpy	as 		np
import 	time
import  random
import	torch
import	matplotlib.pyplot as plt



dtype		=	torch.FloatTensor
if(torch.cuda.is_available()): dtype = torch.cuda.FloatTensor



def get_run_parameters(SB):
	#GET MISCELANEOUS RUN PARAMETERS
	SB['start_time']	= 	writer.start_time;			
	SB['run_path']		= 	writer.run_path
	SB['num_threads']	=	get_num_threads();			
	SB['cuda_avail']	=	cuda.is_available()
	SB['dtype']			=	dtype
	# if(cuda.is_available()): SB['dtype'] = cuda.FloatTensor

	writer.log("RUN PARAMETERS:");							
	writer.log_dict(SB)
	#return param

def get_defaults(SB):
	defaults={}
	defaults['rmse_tol'] 		=	10.0*10**(-8.0)		#STOP FITTING WHEN TRAINING RMSE CHANGE IS LESS THAN RMSE_DELTA
	defaults['rmse_final'] 		=	0.0					#STOP FITTING WHEN TRAINING RMSE HAS REACHED THIS VALUE 
	defaults['rmse_max'] 		=	10.0				#EXIT RUN IF RMSE EXCEEDES THIS (i.e DIVERGES) 
	defaults['i_save_every'] 	=	250					#WRITE FILES EVERY THIS MANY STEPS 
	defaults['write_lsp'] 		=	False	
	defaults['train_edges'] 	=	True		
	defaults['n_rand_GIDS'] 	=	0					#RANDOMLY GRAB n_rand_GIDS FOR test_set IF test_set=[]					 
				 
	defaults['fraction_train'] 	=	1.0		
	#defaults['expected_keys']	=	['nn_file_path', 'dataset_path', 'train_forces', 'max_iter', 'l2_reg_param', 'e_shift', 'learning_rate']
	writer.log("DEFAULT PARAMETERS: (NOTE: see src/util.py to modify defaults)");			
	writer.log_dict(defaults)
	SB.update(defaults)


def dump_poscars(SB):
	for structure in SB['structures'].values(): 
		writer.write_poscar(structure)
		#structure.compute_nbl_for_structure(SB); 

#COMPUTE NBL 
def compute_all_nbls(SB):
	writer.log(["COMPUTING NEIGHBOR LIST (NBL):"])
	start = time.time();	
	for structure in SB['structures'].values():  structure.compute_nbl(SB); 
	writer.log(["	NBL CONSTRUCTION TIME (SEC)	=",time.time()-start])

def compute_all_lsps(SB):
	writer.log(["COMPUTING LOCAL STRUCTURE PARAMETERS (LSP):"])
	start = time.time();	
	for structure in SB['structures'].values():  structure.compute_lsp(SB); 
	writer.log(["	LSP CONSTRUCTION TIME (SEC)	=",time.time()-start])


def partition_data(SB):

	output={}
	test_SIDS=[] 	
	training_SIDS=[] 	
	validation_SIDS=[] 	

	writer.log("PARTITIONING DATA:")

	fraction_train=SB['fraction_train']
	train_edges=SB['train_edges']

	#ERROR CHECKS 
	if(fraction_train==0): 						raise ValueError("FRACTION_TRAIN=0 (CANT TRAIN WITHOUT TRAINING DATA)");
	# if(fraction_train==1): 						raise ValueError("FRACTION_TRAIN=1 (PLEASE CHOOSE 0<FRACTION_TRAIN<1)");
	if(fraction_train<0 or fraction_train>1): 	raise ValueError("BAD VALUE FOR FRACTION_TRAIN: (I.E. FRACTION_TRAIN<0 OR FRACTION_TRAIN>1)");
	if(SB['n_rand_GIDS']>=len(SB['group_sids'].keys())): 	
			raise ValueError("N_RAND_GIDS IS LARGER THAN TOTAL NUMBER OF GIDS: USE N_RAND_GIDS<"+str(len(SB['group_sids'].keys())));


	writer.log(["	TOTAL NUMBER OF CONFIGS	: ",SB['n_structs']])

	#-------------------------------------
	#TEST-SET (EXTRAPOLATION)
	#-------------------------------------
	#RANDOMLY FORM TEST SET IF DESIRED 
	if(SB['test_set']==[] and SB['n_rand_GIDS']!=0):
			k=1
			while(k<=SB['n_rand_GIDS']):
				rand_GID=random.choice(list(SB['group_sids'].keys()))
				if(rand_GID not in SB['test_set'] ):
					SB['test_set'].append(rand_GID)
					k=k+1

	writer.log("	TEST SET (UNTRAINED):")

	for GID in SB['group_sids'].keys(): 
		if(GID in SB['test_set']): 
			writer.log(["		GID		: ",GID])
			for SID in SB['group_sids'][GID]:
				test_SIDS.append(SID)
	remainder=[] #whats left (use for train+val)
	for SID in SB['structures'].keys(): 
		if(SID not in test_SIDS):
			remainder.append(SID)
	# if(test_SIDS==[]): test_SIDS.append(0) #needs to have at least one structure

	#-------------------------------------
	#TRAIN-VALIDATION-SET (TRAIN+INTERPOLATION SET)
	#-------------------------------------
	#TRAINING SIDS (LIST OF DICTIONARY KEYS)a
	train_indices=np.random.choice(len(remainder),int(fraction_train*len(remainder)), replace=False).tolist() #keys for training structures
	for i in train_indices: 
		training_SIDS.append(remainder[i])

	# #ADD MIN/MAX VOLUME STRUCTURES IN EACH GROUP TO TRAINING SET
	if(train_edges):
		sid_2_add=[]  
		for i in SB['group_sids'].values(): sid_2_add.append(i[0]);  sid_2_add.append(i[-1])  #already sorted by volume 
		#print(sid_2_add)
		for SID in sid_2_add: 
			if(SID not in training_SIDS and SID not in test_SIDS): training_SIDS.append(SID)

	# #VALIDATION SID
	for SID in remainder: 
		if(SID not in training_SIDS): validation_SIDS.append(SID)

	if(SB['n_structs'] != len(training_SIDS)+len(validation_SIDS)+len(test_SIDS)):
		raise ValueError("LOST A STUCTURE IN DATA PARTITIONING");

	if(test_SIDS==[]): test_SIDS=validation_SIDS #not ideal but test_SIDS cant be empty
	writer.log(["	N_train_structures	: ",len(training_SIDS)])
	writer.log(["	N_val_structures	: ",len(validation_SIDS)])
	writer.log(["	N_test_structures	: ",len(test_SIDS)])
	writer.log(["	N_combined		: ",len(training_SIDS)+len(validation_SIDS)+len(test_SIDS)])

	SB['test_SIDS']=test_SIDS
	SB['validation_SIDS']=validation_SIDS
	SB['training_SIDS']=training_SIDS

def construct_matrices(SB):

	if(SB['pot_type'] != "NN"): raise ValueError("REQUESTED MODEL NOT CODED YET");


	if(SB['pot_type'] == "NN"):

		U1=[]; N1=[]; V1=[]; Gis=[] #DFT
		Na=0
		for i in SB['training_SIDS']: 	Na=Na+SB['structures'][i].N

		# print(SB['training_SIDS'])
		R1=torch.zeros(len(SB['training_SIDS']),int(Na)).type(dtype); #REDUCTION TENSOR 
		j=0; k=0
		for i in SB['training_SIDS']:
				# print(i)	
				# print(SB['structures'][i].U)
				U1.append(SB['structures'][i].U)
				N1.append(SB['structures'][i].N)
				V1.append(SB['structures'][i].v)

				for Gi in SB['structures'][i].lsps:
					Gis.append(Gi)
					R1[j][k]=1
					k=k+1
				j=j+1
				Na=Na+SB['structures'][i].N

		Gis=torch.tensor(Gis).type(dtype);
		nn_out=SB['nn'].ANN(Gis)
		U1=torch.tensor(np.transpose([U1])).type(dtype);
		N1=torch.tensor(np.transpose([N1])).type(dtype);


		U2=R1.mm(nn_out)
		u2=U2/N1
		u1=U1/N1

		print(Gis.shape,nn_out.shape,R1.shape,U1.shape,U2.shape)

		RMSE=(((u1-u2)**2.0).sum()/len(u1))**0.5
		print(RMSE)

		#for i in range(0,len(V1)):
		#	print(V1[i],u1[i].item(), u2[i].item())
			# #print(R1.shape) 
			# #exit()
			# y1=[]; k=0; Ninv1=[];  i=0; grp_weighs1=[]; grp_weighs2=[]; x1=[];
			# if(FAST_FIT != True):  
			# 	V_N=[]; v11=[]; SID11=[];  SID1=[]; GID=[];
			# for i1 in bin_j2: #loop over structures
			# 	#print(bin_j2); exit()
			# 	if(FAST_FIT): ID_KEY.append(i1)
			# 	y1.append(structs[i1][1]) #vector of structure energies
			# 	V_N.append(structs[i1][3])
			# 	SID1.append(structs[i1][4])
			# 	GID.append(structs[i1][2])
			# 	Ninv1.append(1.0/structs[i1][0]) #vector of 1/N for each structure

				#if(structs[i1][2] 


