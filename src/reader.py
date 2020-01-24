#Author: James hickman 
#Description: As the name implies this code reads the various files needed for pyfit to run
from 	json		import		load
from  	os 		import  	path 
import  writer		 
from 	numpy		import 		array,random
from	classes		import		Neural_Network,Structure
import 	util

def read_input(SB): 
	file_path=SB['input_file']
	if(path.exists(file_path)):
		writer.log("READING INPUT PARAMETERS IN JSON FILE:");
		with open(file_path, "r") as read_file:		#READ USER INPUT FILE
			input_data = load(read_file)
		writer.log_dict(input_data);	SB.update(input_data) 
	else:
		raise ValueError("INPUT_FILE="+str(file_path)+" DOESNT EXIST")

	# #CHECK THAT INPUT FILE HAS THE CORRECT STUFF
	# if(sorted(data.keys()) != sorted(expected_keys)):
	# 	error  = "INPUT FILE APPEARS INCORRECT (I.E. MISSING OR EXTRA JSON KEYS)\n"
	# 	error += "EXPECTED INPUTS:"; 
	# 	for i in expected_keys: error += "\n \t"+i
	# 	raise ValueError(error)
#	# return data

def read_nn_file(SB): 
	file_path=SB['nn_file_path'];
	writer.log("READING NEURAL NETWORK FILE:")
	if(path.exists(file_path)):
		input_file = open( file_path, "r" );  lines=[]
		for line in input_file:
			parts=line.strip().split() 			#rm /n 
			if(len(parts)==1):	parts=parts[0]	#dont save single numbers as arrays
			if(parts!=[]): lines.append(parts)
		nn=Neural_Network(lines) 				#send lines to NN class to create NN object
	else: 
		raise ValueError("NN_FILE="+str(file_path)+" DOESNT EXIST")
	writer.log_dict(nn.info);
	SB.update(nn.info)
	SB['nn']=nn					
	return nn

def read_database(SB): 
	file_path=SB['dataset_path']
	writer.log("READING DATASET FILE:");  										\
	#returns a list of objects of Class Structure
	structures={};  SID=0;  		N_atoms=0; new_structure=True;
	group_sids={};  group_info={}; 	#collect information for various structural groups
	if(path.exists(file_path)):
		input_file = open( file_path, "r" )
		for line in input_file:
			if(new_structure):
				lines=[];  new_structure=False;  counter=1; 
			else:
				counter += 1
			parts=line.strip().split() 
			if(len(parts)==1): parts=parts[0]			#dont save single numbers as arrays
			lines.append(parts)
			#TODO: THIS HARDCODE FOR CURRENT POSCAR FORMAT (NEED TO GENERALIZE)
			if(counter==6): Natom=int(parts);
			if(counter>6):
				if(counter==6+1+Natom+1): 				#see dataset examples for format
					N_atoms+=Natom
					structures[SID]=Structure(lines,SID,SB); 	#send lines to Structure class to create object
					GID=str(structures[SID].comment)	#name of structural groups '%35s'%
					if( GID not in group_sids.keys()):
						group_sids[GID]=[]
						group_info[GID]=[Natom] 
					group_sids[GID].append([structures[SID].v,SID])
					new_structure=True;  SID += 1 ; #print(lines)
	else: 
		raise ValueError("DATABASE_FILE="+str(file_path)+" DOESNT EXIST")	
	for i in group_info.keys():
		group_sids[i]=[item[1] for item in sorted(group_sids[i])]	#sort group SID by increasing volume
		group_info[i].append(len(group_sids[i])) 					#number of structures in the group
		group_info[i].append(group_info[i][0]*group_info[i][1]) 	#total number of  in the group

	dataset_info={};  
	dataset_info["n_structs"]=len(structures.keys())	
	dataset_info["n_atoms"]=N_atoms; 
	writer.log_dict(dataset_info);	
	writer.write_group_summary(group_info)

	#UPDATE SB DICT
	SB.update(dataset_info)
	SB['group_info']=group_info
	SB['group_sids']=group_sids
	SB['structures']=structures

	if(SB['dump_poscars']):
		util.dump_poscars(SB)			 


 
	# return [dataset_info,group_sids,group_info,structures]
