#Author: James hickman 
#Description: As the name implies this code reads the various files needed for pyfit to run
from 	json	import	load
from  	os 	import  path 
import  writer		 
from 	numpy	import 	array,random

import 	neural
import  data
from	torch	import cuda,

def read_input(SB): 

	#READ DEFAULT FILE
	file_path=SB['src_path']+'/defaults.json'
	if(path.exists(file_path)):
		writer.log(["READING DEFAULT PARAMETERS USING FILE:",file_path]);
		with open(file_path, "r") as read_file:		#READ USER INPUT FILE
			input_data = load(read_file)
		writer.log_dict(input_data);	
		SB.update(input_data) 
	else:
		raise ValueError("DEFAULT_FILE="+str(file_path)+" DOESNT EXIST")

	#READ INPUT FILE	
	file_path=SB['input_file']
	if(path.exists(file_path)):
		writer.log(["OVERWRITING SELECT DEFAULTS USING INPUT FILE:",SB['input_file']]);
		with open(file_path, "r") as read_file:		#READ USER INPUT FILE
	 		input_data = load(read_file)
		writer.log_dict(input_data);	
		SB.update(input_data)
	else:
		raise ValueError("INPUT_FILE="+str(file_path)+" DOESNT EXIST")

	if(SB['use_cuda'] and  cuda.is_available()==True ):
		writer.log("NOTE: CUDA IS NOT AVAILABLE (RE-SETTING)")
		writer.log("	use_cuda		:	False")
		SB['use_cuda']=False


	exit()
	#USING FLAG FROM INPUT FILE DEFINE DTYPE
	writer.log("CUDA INFORMATION:");
	dtype=FloatTensor; TMP={}
	TMP['cuda_avail']=cuda.is_available() 
	TMP['use_cuda']	 =SB['use_cuda']
	if(TMP['cuda_avail'] and SB['use_cuda']): dtype = cuda.FloatTensor
	TMP['dtype']=dtype
	writer.log_dict(TMP);	
	SB.update(TMP)





	if('pot_type' not in SB.keys() or 'pot_file' not in SB.keys()  
		or 'dataset_path' not in SB.keys()):
		raise ValueError("INPUT FILE MUST CONTAIN KEYS FOLLOWING KEYS: pot_type, pot_file, dataset_path")
	if(SB['pot_type'] != "NN"):
		raise ValueError("REQUESTED POT_TYPE="+str(SB['pot_type'])+" IS NOT AVAILABLE")



def read_pot_file(SB): 
	if(SB['pot_type']=="NN"):
		file_path=SB['pot_file'];
		writer.log("READING NEURAL NETWORK FILE:")
		if(path.exists(file_path)):
			input_file = open( file_path, "r" );  lines=[]
			for line in input_file:
				parts=line.strip().split() 			#rm /n 
				if(len(parts)==1):	parts=parts[0]	#dont save single numbers as arrays
				if(parts!=[]): lines.append(parts)
			pot=neural.NN(lines,SB) 				#send lines to NN class to create NN object
		else: 
			raise ValueError("NN_FILE="+str(file_path)+" DOESNT EXIST")
		writer.log_dict(pot.info);
		SB.update(pot.info)
		SB['nn']=pot					
	#return nn


def read_database(SB): 
	file_path=SB['dataset_path']
	writer.log("READING DATASET FILE:");  		 

	full_set=data.Dataset("full") #INITIALIZE FULL DATASET OBJECT

	SID=0;  new_structure=True;
	if(path.exists(file_path)):
		input_file = open( file_path, "r" )
		for line in input_file:
			if(new_structure):
				lines=[];  new_structure=False;  counter=1;
				full_set.Ns+=1 
			else:
				counter += 1
			parts=line.strip().split() 
			if(len(parts)==1): parts=parts[0]			#dont save single numbers as arrays
			lines.append(parts)
			#TODO: THIS HARDCODE FOR CURRENT POSCAR FORMAT (NEED TO GENERALIZE)
			if(counter==6): Natom=int(parts);
			if(counter>6):
				if(counter==6+1+Natom+1): 				#see dataset examples for format
					full_set.Na+=Natom
					full_set.structures[SID]=data.Structure(lines,SID,SB); 	#create structure object
					GID=str(full_set.structures[SID].gid)	
					if( GID not in full_set.group_sids.keys()):
						full_set.group_sids[GID]=[]
					full_set.group_sids[GID].append([full_set.structures[SID].v,SID])
					new_structure=True;  SID += 1 ; #print(lines)
	else: 
		raise ValueError("DATABASE_FILE="+str(file_path)+" DOESNT EXIST")	

	full_set.sort_group_sids()
	writer.log(["	TOTAL NUMBER OF STRUCTURES:",full_set.Ns])
	writer.log(["	TOTAL NUMBER OF ATOMS:	",full_set.Na])
	SB['full_set']=full_set

