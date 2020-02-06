import  writer
import	numpy	as 	np
import 	time
import  random
from 	os	import  path 
import  data

def get_run_parameters(SB):
	#GET MISCELANEOUS RUN PARAMETERS
	SB['src_path']		= 	path.dirname(path.realpath(__file__))
	SB['run_path']		= 	writer.run_path	
	SB['start_time']	= 	writer.start_time;	
	writer.log("RUN PARAMETERS:");							
	writer.log_dict(SB)

def dump_poscars(SB):
	for structure in SB['full_set'].structures.values(): 
		writer.write_poscar(structure)
		#structure.compute_nbl_for_structure(SB); 

def compute_all_nbls(SB):
	writer.log(["COMPUTING NEIGHBOR LIST (NBL):"])
	start = time.time();	
	for structure in SB['full_set'].structures.values():  
		structure.compute_nbl(SB); 
	writer.log(["	NBL CONSTRUCTION TIME (SEC)	=",time.time()-start])

def compute_all_lsps(SB):
	writer.log(["COMPUTING LOCAL STRUCTURE PARAMETERS (LSP):"])
	start = time.time();	
	for structure in SB['full_set'].structures.values():  
		structure.compute_lsp(SB); 
	writer.log(["	LSP CONSTRUCTION TIME (SEC)	=",time.time()-start])

def chkpnt(SB,t):
	if(SB['pot_type']=='NN'):
		SB['nn'].unset_grad()
		writer.write_NN(SB['nn'],t) 
		for ds_name in SB['datasets']:
			SB[ds_name].report(SB,t)
		SB['nn'].set_grad()

def partition_data(SB):
	test_set=data.Dataset("test",SB) #INITIALIZE FULL DATASET OBJECT
	training_set=data.Dataset("train",SB) #INITIALIZE FULL DATASET OBJECT
	validation_set=data.Dataset("validate",SB) #INITIALIZE FULL DATASET OBJECT

	writer.log("PARTITIONING DATA:")
	writer.log(["	TOTAL NUMBER OF GROUPS=",len(SB['full_set'].group_sids.keys())])
	fraction_train=SB['fraction_train']
	train_edges=SB['train_edges']

	#ERROR CHECKS 
	if(fraction_train==0): 						
		raise ValueError("FRACTION_TRAIN=0 (CANT TRAIN WITHOUT TRAINING DATA)");
	if(fraction_train<0 or fraction_train>1): 
		ERR="BAD VALUE FOR FRACTION_TRAIN: (I.E. FRACTION_TRAIN<0 OR FRACTION_TRAIN>1)"	
		raise ValueError(ERR);
	if(SB['n_rand_GIDS']>=len(SB['full_set'].group_sids.keys())): 	
			ERR="N_RAND_GIDS IS LARGER THAN TOTAL NUMBER OF GIDS: USE N_RAND_GIDS<"  \
			+str(len(SB['full_set'].group_sids.keys()))
			raise ValueError(ERR);

	#-------------------------------------
	#TEST-SET (EXTRAPOLATION)
	#-------------------------------------

	if(SB['fix_rand_seed']): random.seed(a=412122, version=2)
					
	#RANDOMLY FORM TEST SET IF DESIRED 
	if(SB['test_set_gids']==[] and SB['n_rand_GIDS']!=0):
			k=1
			while(k<=SB['n_rand_GIDS']):
				rand_GID=random.choice(list(SB['full_set'].group_sids.keys()))
				#if(rand_GID not in SB['test_set_gids'] ):
				for i1 in SB['exclude_from_test']:
					if(i1 not in rand_GID): 
						keep=True
					else:
						keep=False; break

				if(rand_GID not in SB['test_set_gids'] and keep): #REMOVE
					writer.log("	"+rand_GID)
					SB['test_set_gids'].append(rand_GID)
					k=k+1
	#exit()
	writer.log("	TEST SET (UNTRAINED):")

	for GID in SB['full_set'].group_sids.keys(): 
		if(GID in SB['test_set_gids']): 
			writer.log(["		GID		: ",GID])
			#test_set.group_sids[GID]= SB['full_set'].group_sids[GID] 
			for SID in SB['full_set'].group_sids[GID]:
				test_set.structures[SID]= SB['full_set'].structures[SID]
				test_set.Ns+=1;		test_set.Na+=SB['full_set'].structures[SID].N

	remainder=[] #whats left (use for train+val)

	for SID in SB['full_set'].structures.keys(): 
		if(SID not in test_set.structures.keys()):
			remainder.append(SID)

	#-------------------------------------
	#TRAIN-VALIDATION-SET (TRAIN+INTERPOLATION SET)
	#-------------------------------------
	#TRAINING SIDS (LIST OF DICTIONARY KEYS)a
	train_indices=np.random.choice(len(remainder),int(fraction_train*len(remainder)), replace=False).tolist() #keys for training structures
	for i in train_indices: 
		training_set.structures[remainder[i]]= SB['full_set'].structures[remainder[i]] 
		training_set.Ns+=1;		
		training_set.Na+=SB['full_set'].structures[remainder[i]].N

	# #ADD MIN/MAX VOLUME STRUCTURES IN EACH GROUP TO TRAINING SET
	if(train_edges):
		sid_2_add=[]  
		for i in SB['full_set'].group_sids.values():
			if(len(i)>4): #already sorted by volume 
				sid_2_add.append(i[0]);   sid_2_add.append(i[1])  
				sid_2_add.append(i[-2]);  sid_2_add.append(i[-1])   
		#print(sid_2_add)
		for SID in sid_2_add: 
			if(SID not in training_set.structures.keys() and SID not in test_set.structures.keys()):  
				training_set.structures[SID]= SB['full_set'].structures[SID] 
				training_set.Ns+=1;		
				training_set.Na+=SB['full_set'].structures[SID].N

	# #VALIDATION SID
	for SID in remainder: 
		if(SID not in training_set.structures.keys()):  
			validation_set.structures[SID]= SB['full_set'].structures[SID] 
			validation_set.Ns+=1;		
			validation_set.Na+=SB['full_set'].structures[SID].N

	if(SB['full_set'].Ns != training_set.Ns+test_set.Ns+validation_set.Ns):
		raise ValueError("LOST A STUCTURE IN DATA PARTITIONING");

	# if(test_SIDS==[]): test_SIDS=validation_SIDS #not ideal but test_SIDS cant be empty
	writer.log(["	N_train_structures	: ",training_set.Ns])
	writer.log(["	N_val_structures	: ",validation_set.Ns])
	writer.log(["	N_test_structures	: ",test_set.Ns])
	writer.log(["	N_combined		: ",training_set.Ns+test_set.Ns+validation_set.Ns])

	test_set.build_arrays(SB)
	training_set.build_arrays(SB)
	validation_set.build_arrays(SB)

	SB['test_set']=test_set
	SB['training_set']=training_set
	SB['validation_set']=validation_set
	SB['datasets']=['test_set','training_set','validation_set']
