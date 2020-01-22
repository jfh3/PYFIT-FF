import  writer
from 	torch	import	get_num_threads,cuda
import	numpy	as 		np
import 	time

import matplotlib.pyplot as plt


def get_run_parameters(params):
	#GET MISCELANEOUS RUN PARAMETERS
	params['start_time']	= 	writer.start_time;			
	params['run_path']	= 	writer.run_path
	params['num_threads']	=	get_num_threads();			
	params['cuda_avail']	=	cuda.is_available()
	writer.log("RUN PARAMETERS:");							
	writer.log_dict(params)
	#return param

def get_defaults(params):
	defaults={}
	defaults['rmse_tol'] 		=	10.0*10**(-8.0)				#STOP FITTING WHEN TRAINING RMSE CHANGE IS LESS THAN RMSE_DELTA
	defaults['rmse_final'] 		=	0.0					#STOP FITTING WHEN TRAINING RMSE HAS REACHED THIS VALUE 
	defaults['rmse_max'] 		=	10.0					#EXIT RUN IF RMSE EXCEEDES THIS (i.e DIVERGES) 
	defaults['i_save_every'] 	=	250					#WRITE FILES EVERY THIS MANY STEPS 
	defaults['write_lsp'] 		=	False	
	defaults['train_edges'] 	=	True						 
	defaults['fraction_train'] 	=	0.8						 
	#defaults['expected_keys']	=	['nn_file_path', 'dataset_path', 'train_forces', 'max_iter', 'l2_reg_param', 'e_shift', 'learning_rate']
	writer.log("DEFAULT PARAMETERS: (NOTE: see src/util.py to modify defaults)");			
	writer.log_dict(defaults)
	params.update(defaults)


def dump_poscars(params):
	for structure in params['structures'].values(): 
		writer.write_poscar(structure)
		#structure.compute_nbl_for_structure(params); 

#COMPUTE NBL 
def compute_all_nbls(params):
	writer.log(["COMPUTING NEIGHBOR LIST (NBL): (ESTIMATED TIME IN SECONDS)	=",0.0000977686*params['n_atoms']])
	start = time.time();	
	for structure in params['structures'].values():  structure.compute_nbl_for_structure(params); 
	writer.log(["	NBL CONSTRUCTION TIME (SEC)	=",time.time()-start])

def compute_all_lsps(params):
	writer.log(["COMPUTING LOCAL STRUCTURE PARAMETERS (LSP): (ESTIMATED TIME IN SECONDS)	=",0.00047787*params['n_atoms']])
	start = time.time();	
	for structure in params['structures'].values():  structure.compute_lsp_for_structure(params); 
	writer.log(["	LSP CONSTRUCTION TIME (SEC)	=",time.time()-start])

def split_data_train_validation(params):
	fraction_train=params['FRACTION_TRAIN']
	train_edges=params['TRAIN_EDGES']
	n_structures=params['n_structs']

	writer.log("PARTITIONING DATA:")
	writer.log(["	TOTAL NUMBER OF STRUCTURES	=",n_structures])
	if(fraction_train==0): 						raise ValueError("fraction_train=0 (cant train without training data)");
	if(fraction_train==1): 						raise ValueError("fraction_train=1 (please choose 0<fraction_train<1)");
	if(fraction_train<0 or fraction_train>1): 	raise ValueError("bad value for fraction_train: (i.e. fraction_train<0 or fraction_train>1)");

	#TRAINING SIDS (LIST OF DICTIONARY KEYS)a
	training_structures=np.random.choice(n_structures,int(fraction_train*n_structures), replace=False).tolist() #keys for training structures

	#ADD MIN/MAX VOLUME STRUCTURES IN EACH GROUP TO TRAINING SET
	if(train_edges):
		sid_2_add=[]  
		for i in params['group_sids'].values(): sid_2_add.append(i[0]);  sid_2_add.append(i[-1])  #already sorted by volume 
		for i in params['structures'].keys(): 
			if(i in sid_2_add and i not in training_structures): training_structures.append(i)

	#VALIDATION SID
	validation_structures=[];
	for i in params['structures'].keys(): 
		if(i not in training_structures): validation_structures.append(i)

	writer.log(["	NUMBER OF TRAINING STRUCTURES	=",len(training_structures)])
	writer.log(["	NUMBER OF VALIDATION STRUCTURES	=",len(validation_structures)])

	if(len(training_structures)+len(validation_structures)!=n_structures): 	raise ValueError("LOST STRUCTURE DURING DATA PARTITION") 

	params['training_structures']=training_structures
	params['validation_structures']=validation_structures


#"force_train_list"		:	["0008-DC-C-PERTURBED+ISO+SMALL","0009-DC-C-PERTURBED+ISO+TINY","0010-DC-C-PERTURBED+SINGLE-ATOM","0007-DC-C-PERTURBED+ISO","0041-LIQUID-PERTURBED+ISO"],


def force_time_estimate(params):
	print('IF WE ONLY TRAIN ENERGY:')
	print("	number of structures=",params['n_structs'])
	print("	number of atoms=",params['n_atoms'])
	Nstructure_new=0; Natom_new=0
	N1=0
	for i in params['structures'].values():
		if(i.train_forces):
			N1=N1+1
			Nstructure_new += 3*i.N
			Natom_new += 3*i.N*i.N
	print('IF WE ONLY TRAIN FORCES AS REQUESTED:')
	print("	number of structures for which force training will be done",N1)
	print("	number of perturbed structures",Nstructure_new)
	print("	number of added atoms=",Natom_new*0.3)

	# print("	number of structures for which force training will be done",N1*0.25)
	# print("	number of perturbed structures",Nstructure_new*0.25)
	# print("	number of added atoms=",Natom_new*0.25)


	exit()



# writer.log(["", *n_atoms])
# start = time.time()
# for structure in structures.values(): 
# 	structure.Compute_LSP(nn.info) 
# writer.log(["",time.time()-start])


# def get_run_parameters():
# 	#GET MISCELANEOUS RUN PARAMETERS
# 	param={}
# 	param['start_time']		= 	writer.start_time;			param['run_path']	= 	writer.run_path
# 	param['num_threads']	=	get_num_threads();			param['cuda_avail']	=	cuda.is_available()
# 	writer.log("RUN PARAMETERS:");							writer.log_dict(param)
# 	defaults=config.report_variables()
# 	writer.log("DEFAULT PARAMETERS:");						writer.log_dict(defaults)
# 	param.update(defaults)
# 	return param



		 

# locals().update(get_defaults())

# def find_outliers_IQR(params):
# 	for gid in params['group_sids'].keys():
# 		e=[]; v=[]
# 		for sid in params['group_sids'][gid]: 
# 			e.append(params['structures'][sid].E/params['structures'][sid].N)
# 			v.append(params['structures'][sid].V/params['structures'][sid].N)
# 		e=np.array(e); 		v=np.array(v)
# 		v75, v25 = np.percentile(v, [75 ,25])
# 		e75, e25 = np.percentile(e, [75 ,25])
# 		iqre=e75-e25; 		iqrv=v75-v25

# 		x1=[]; y1=[]
# 		x2=[]; y2=[]
# 		range_length=1.0
# 		for i in range(0,len(v)):
# 			if(e[i]>e25-range_length*iqre and e[i]<e75+range_length*iqre and v[i]>v25-range_length*iqrv and v[i]<v75+range_length*iqrv):
# 				x1.append(v[i]); y1.append(e[i])
# 			else:
# 				x2.append(v[i]); y2.append(e[i])

# 		plt.plot(x1, y1,'.',x2, y2,'o')
# 		plt.show()


# def find_outliers_std(params):
# 	for gid in params['group_sids'].keys():
# 		e=[]; v=[]
# 		for sid in params['group_sids'][gid]: 
# 			e.append(params['structures'][sid].E/params['structures'][sid].N)
# 			v.append(params['structures'][sid].V/params['structures'][sid].N)
# 		e=np.array(e); 		v=np.array(v)
# 		e_std=np.std(e);	v_std=np.std(v)
# 		e_m=np.mean(e);		v_m=np.mean(v)

# 		x1=[]; y1=[]
# 		x2=[]; y2=[]
# 		N_std=1.5
# 		for i in range(0,len(v)):
# 			if(e[i]>e_m-N_std*e_std and e[i]<e_m+N_std*e_std and v[i]>v_m-N_std*v_std and v[i]<v_m+N_std*v_std):
# 				x1.append(v[i]); y1.append(e[i])
# 			else:
# 				x2.append(v[i]); y2.append(e[i])

# 		plt.plot(x1, y1,'.',x2, y2,'o')
# 		plt.show()
# 		#exit()

# # ax.plot(t, s)

# def find_outliers_2(params):
# 	# x1=[]; y1=[]
# 	# x2=[]; y2=[]
# 	for gid in params['group_sids'].keys():
# 		data=[]
# 		data2=[]
# 		print(gid)

# 		for sid in params['group_sids'][gid]: 
# 			structure=params['structures'][sid]
# 			first=True
# 			for lsp in structure.lsps:
# 				if(first):
# 					tmp=lsp; first=False
# 				else:
# 					tmp=np.concatenate((tmp,lsp))

# 			data.append(tmp)
# 			data2.append([structure.V/structure.N,structure.E/structure.N])
# 		#print(structure.N)

# 		data=np.array(data)
# 		#print(data.shape); #exit()
# 		clf = IsolationForest( behaviour = 'new', max_samples='auto', contamination= 'auto')
# 		#NOTE: contamination is an important parameter which controls now dramatically outliers
# 		#     are selectect (0=no outliers --> larger values yields more aggressive selection. 'auto' is a good choice)  
# 		preds = clf.fit_predict(data)
# 		x1=[]; y1=[]
# 		x2=[]; y2=[]
# 		#print(preds)
# 		for i,v in enumerate(preds):
# 			# structure=params['structures'][sid]

# 			if(v==1):
# 				x1.append(data2[i][0]); y1.append(data2[i][1])
# 			else: 
# 				x2.append(data2[i][0]); y2.append(data2[i][1])

# 		plt.plot(x1, y1,'.',x2, y2,'o')
# 		plt.show()

# 		# print(data.shape)

