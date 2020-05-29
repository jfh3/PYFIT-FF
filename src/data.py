import	numpy	as	np
import  torch
import	writer

fit_diff=False

class Dataset:
	def __init__(self,name,SB):
		self.name=name
		self.structures={};  #structure[SID]  --> structure_objects
		self.group_sids={}   #group_sids[GID] --> list of SID
		self.Na=0;
		self.Ns=0;

		self.dtype=torch.FloatTensor
		self.dtype2=torch.LongTensor
		if(SB['use_cuda'] and self.name=="train"): 
			self.dtype=torch.cuda.FloatTensor
			self.dtype2=torch.cuda.LongTensor

	def sort_group_sids(self): 
		for i in self.group_sids.keys():
			self.group_sids[i]=[item[1] for item in sorted(self.group_sids[i])]	 

	def build_arrays(self,SB): 

		if(self.Ns!=0):
		
			#ALL MODELS NEED THE FOLLOWING 
			u1=[]; v1=[]; N1=[]; self.SIDS1=[]; self.GIDS1=[];	#DFT STUFF
			swt1=[];  swt2=[]; mask2=[]; j=0

			#GET MAX NUMBER OF ATOMS IN DATASET
			for structure in self.structures.values():	N1.append(structure.N)
			self.Nmax=max(N1);	self.mask=[]

			for structure in self.structures.values():
					u1.append(structure.u)
					v1.append(structure.v)
					swt1.append(structure.weight1)
					swt2.append(structure.weight2)
					if(structure.weight2!=0): mask2.append(j)
					j=j+1

					self.SIDS1.append(structure.sid)
					self.GIDS1.append(structure.gid)

					#BUILD MASK FOR REDUCTION
					counter=1
					while(counter<=self.Nmax):
						if(counter<=structure.N):
							self.mask.append([True])
						else:
							self.mask.append([False])
						counter += 1

			self.mask=torch.tensor(self.mask)			
			self.v1=torch.tensor(np.transpose([v1])).type(self.dtype);
			self.u1=torch.tensor(np.transpose([u1])).type(self.dtype);
			self.u2=torch.tensor(np.transpose([u1])).type(self.dtype); #initialize
			self.N1=torch.tensor(np.transpose([N1])).type(self.dtype);
			self.swt1=torch.tensor(np.transpose([swt1])).type(self.dtype);	#FOR RMSE

			#FOR DIFF
			self.mask2 = mask2
			self.ud1=self.u1[self.mask2]
			self.swt2=(torch.tensor(np.transpose([swt2])).type(self.dtype))[self.mask2]; 	

			if(SB['pot_type'] == "NN"):
				Gis=[] 
				for structure in self.structures.values():
					for Gi in structure.lsps:	Gis.append(Gi)
				self.Gis=torch.tensor(Gis).type(self.dtype);
				self.M1=torch.tensor([np.ones(len(self.Gis))]).type(self.dtype)

	def report(self,SB,t):
		if(self.Ns!=0):
			if(self.name!="train" and  SB['use_cuda']):  SB['nn'].send_to_cpu()
			self.evaluate_model(SB)
			if(self.name!="train" and  SB['use_cuda']):  SB['nn'].send_to_gpu()

			if(t%SB['save_every2']==0  or t==111111): writer.write_E_vs_V(self,t)

			RMSE=(((self.u1-self.u2)**2.0).sum()/self.Ns)**0.5 
			MAE=(((self.u1-self.u2)**2.0)**0.5).sum()/self.Ns
			MED_AE=torch.median((((self.u1-self.u2)**2.0)**0.5))
			STD_AE=torch.std(((self.u1-self.u2)**2.0)**0.5) 
			MAX_AE=torch.max(((self.u1-self.u2)**2.0)**0.5) 
			RMS_DU=0.5*((self.u1.view(self.u1.shape[0],1)-self.u1.view(1,self.u1.shape[0])) \
				-(self.u2.view(self.u2.shape[0],1)-self.u2.view(1,self.u2.shape[0]))) 
			RMS_DU=(torch.mean(RMS_DU**2.0)**0.5)

			if(self.name!="no_dft"):
				writer.write_stats(self.name,t,RMSE,MAE,MED_AE,STD_AE,MAX_AE,RMS_DU)


	#APPLY THE MODEL TO DATASET
	def evaluate_model(self,SB):
		#set_name=dateset object name
		if(SB['pot_type']=='NN'):
			
			#EVALUATE NN
			nn_out=SB['nn'].NN_eval(self)

			#PREFORM REDUCTION (SUM(ATOMIC_EN) --> STRUCTURE_EN)
			self.u2=(torch.sum((torch.zeros(self.mask.shape[0], 1)).masked_scatter_(self.mask, nn_out) \
				.view(self.Ns,self.Nmax),1).view(self.Ns,1))/self.N1

	def compute_objective(self,SB):
		global fit_diff 

		self.evaluate_model(SB)

		#OBJECTIVE TERM-1 (RMSE OR MAE)
		err=(self.u1-self.u2) 		#*1000 #convert to meV
		RMSE=(torch.mean(err**2.0)**0.5).item() #NOT USED IN OBJ 
		#if(RMSE<SB['rmse_dU']):
		err=self.swt1*err 			#apply individual structure weights

		OBE1=torch.mean(err**2.0) 
		if(SB['train_RMSE']): OBE1=OBE1**0.5
		OBE1=SB['lambda_E1']*OBE1

		#OBJECTIVE TERM-2 (DIFF)
		OB_DU=torch.tensor(0.0) #.type(SB['dtype'])

		if(RMSE<SB['rmse_dU'] and fit_diff != True):  fit_diff=True

		if(SB['lambda_dU']>0 and fit_diff): 
		#if(SB['lambda_dU']>0 and RMSE<SB['rmse_dU']): 
			#APPLY MASK TO ONLY USE TERMS WITH NON-ZERO WEIGHTS 
			self.ud2=self.u2[self.mask2]
			DIFF1=(self.swt2**0.5)*((self.ud1.view(self.ud1.shape[0],1)-self.ud1.view(1,self.ud1.shape[0])) \
			-(self.ud2.view(self.ud2.shape[0],1)-self.ud2.view(1,self.ud2.shape[0]))) #*1000.0 
			DIFF1=(self.swt2**0.5)*torch.t(DIFF1)
			OB_DU=SB['lambda_dU']*(torch.mean(DIFF1**2.0)**0.5)

		#OBJECTIVE TERMS 3 AND 4:  L1 AND L2 REG ON NN FITTING PARAM
		OBL1=torch.tensor(0.0); OBLP=torch.tensor(0.0)

		if(SB['pot_type']=='NN'):
		
			m1=SB['training_set'].Ns; S=0

			S=0.0; S1=0
			#L1 REGULARIZATION
			if(SB['lambda_L1']>0):
				for i in range(0,len(SB['nn'].submatrices)-1): 
					#NWB=SB['nn'].submatrices[i].shape[0]*SB['nn'].submatrices[i].shape[1]
					#S1=torch.sum(SB['nn'].submatrices[i]**SB['LP'])/NWB
					S1=torch.sum((SB['nn'].submatrices[i]**2.0)**0.5) 
					S+=S1;
				OBL1=SB['lambda_L1']*S/2.0/m1; S=0; S1=0

			#LP P=2-->L2 REGULARIZATION 
			if(SB['lambda_Lp']>0):
				for i in range(0,len(SB['nn'].submatrices)-1): 
					#NWB=SB['nn'].submatrices[i].shape[0]*SB['nn'].submatrices[i].shape[1]
					#S1=torch.sum(SB['nn'].submatrices[i]**SB['LP'])/NWB
					S1=torch.sum(SB['nn'].submatrices[i]**SB['LP']) 
					S+=S1;
				OBLP=SB['lambda_Lp']*S/2.0/m1; S=0;

		return [RMSE,OBE1,OB_DU,OBL1,OBLP]


class Structure:
	def __init__(self,lines,sid,SB):
		#print(lines)
		SF=float(lines[1])	#SCALING FACTOR
		self.sid		= sid
		self.gid		= str(lines[0]) #string describing the group

		self.scale_factor	= SF
		self.a1			= SF*(np.array(lines[2]).astype(np.float))
		self.a2			= SF*(np.array(lines[3]).astype(np.float))
		self.a3			= SF*(np.array(lines[4]).astype(np.float))
		self.V			= np.absolute(np.dot(self.a1,np.cross(self.a2,self.a3)))
		self.N      		= int(lines[5])
		self.U			= float(lines[-1])+self.N*SB['u_shift']  
		
		if(SB['normalize_ei']): 
			raise Exception("ERROR: NORMALIZATION OF ENERGIES IS CURRENTLY DISABLED")
			self.U=self.U/10.0  #change units so energy runs from ~ -1 to 1

		self.v			= self.V/self.N
		self.u			= self.U/self.N
		self.species		= SB['species']  #TODO THIS NEEDS TO BE FIXED (GENERALIZE TO BINARY)

		#INDIVDUAL STRUCTURE WEIGHTS FOR OBJECTIVE FUNCTION 
		self.weight1		= SB['default_weight1'] 	#RMSE WEIGHT
		self.weight2		= SB['default_weight2']		#DIFF WEIGHT
		if(str(type(SB['weight_selector'])) =="<class 'float'>"): #use weight_selector as therehold value
			if(self.u<SB['weight_selector']):
				self.weight1		= SB['mod_weight1']
				self.weight2		= SB['mod_weight2']			
		if(str(type(SB['weight_selector'])) =="<class 'list'>"): #use weight_selector as therehold value
			for tag in SB['weight_selector']:	
				if(tag in self.gid): 
					self.weight1=SB['mod_weight1']; 
					self.weight2=SB['mod_weight2']; 

		if((np.array(lines[7:-1]).astype(np.float)).shape[1] != 3):
			raise Exception("POSCAR FILE HAS MORE THAN 3 ENTRIES ON COORDINATE LINES")

		if(lines[6][0][0] == 'c' or lines[6][0][0] == 'C'): 
			# Nx3 array with positions for atoms
			self.positions	= SF*np.array(lines[7:-1]).astype(np.float)	
		else:
			raise Exception("POSCAR READER CURRENTLY HARDCODED FOR CARTESIAN CASE")

		#UN-INITALIZED BUT FILLED LATER
		self.nbls=[]		#list of neighborlist for each atom (len(nbl_i) isnt the same for all atoms) 
		self.lsps=[]		#list of LSP for each 

	def compute_nbl(self,SB):

		pot_type=SB['pot_type']
		Rc=SB['cutoff_dist']

		#GET LENGTHS OF VECTORS
		a1_n = np.linalg.norm(self.a1)
		a2_n = np.linalg.norm(self.a2)
		a3_n = np.linalg.norm(self.a3)

		#FIND NUMBER OF TIMES TO REPEAT THE LATTICE (do +1 just to be safe)
		x_repeat = int(np.ceil(Rc / a1_n))+1  
		y_repeat = int(np.ceil(Rc / a2_n))+1   
		z_repeat = int(np.ceil(Rc / a3_n))+1   

		#GET CENTER LOCATIONS FOR PERIODIC IMAGES (-x_repeat*a1 to x_repeat*a1). 
		centers=[];
		for i in range(-x_repeat, x_repeat + 1):
			for j in range(-y_repeat, y_repeat + 1):
				for k in range(-z_repeat, z_repeat + 1):
					 # orgin of given periodic "image"
					centers.append(self.a1*i + self.a2*j + self.a3*k)  
		centers=np.array(centers)

		#add each row of positions to each row of centers  
		periodic_structure=(centers+self.positions[:,None,:]).reshape(-1,self.positions.shape[1]) 

		#LOOP OVER ATOMS AND FIND NEIGHBORS WITHIN Rc
		ID=0
		for ri in self.positions:
				#array of distance vectors from atom ri 		
				rij	= periodic_structure - ri 		
				#magnetude of distance vector rij (axis=1-->accross row)					
				rij = np.linalg.norm(rij, axis = 1)		
				#rm i=j and apply cutoff function			
				mask      = (rij > 1e-5) & (rij < Rc)
				neighbors = periodic_structure[mask]
				if(neighbors.shape[0]==0): 
					raise Exception("ATOM HAS NO NEIGHBORS",self.sid,self.gid)

				#convert to difference vectors for neightbors relative to ri
				neighbor_vecs = neighbors - ri 						
				self.nbls.append(neighbor_vecs)
				ID += 1

	def compute_lsp(self,SB):
		rc=SB['cutoff_dist']
		dc4=SB['cutoff_range']**4.0
		ros=np.array(SB['lsp_ro_val']).reshape(len(SB['lsp_ro_val']),1)
		ros2=(ros**2.0)
		lgs=SB['lsp_lg_poly']
		s=SB['lsp_sigma']

		#LOOP OVER NBL FOR EACH ATOM
		for nbl in self.nbls:
			#nbl=np.array([[11,12,13],[21,22,23]]) #np.array([[0,3,0],[2,0,0]])  #useful for testing what the following commands do

			n=nbl.shape[0] #numer of neighbors
			Xij=np.tile(nbl,n).reshape(n*n,3)
			Xik=np.tile(nbl,(n,1)).reshape(n*n,3) #.reshape(3*n,1) #.repeat(n,axis=0).reshape(n*n,3)

			rij=(((Xij**2.0).sum(axis=1))**0.5).reshape(1,n*n); #sum accros row		
			rik=(((Xik**2.0).sum(axis=1))**0.5).reshape(1,n*n);			
			cos_ijk1=((Xij*Xik).sum(axis=1).reshape(1,n*n))/rij/rik;

			#RADIAL TERM
			fcij=(rij-rc)**4.0;  fcij=fcij/(dc4+fcij)
			#mask      =  (rij < rc).astype(np.int);		fcij=fcij*mask		

			fcik=(rik-rc)**4.0;  fcik=fcik/(dc4+fcik)
			#mask      =  (rik < rc).astype(np.int);		fcik=fcik*mask

			if(SB['nn'].info['lsp_type']==5):
				radial_term=np.exp(-((rij-ros)/s)**2.0)*np.exp(-((rik-ros)/s)**2.0)*fcik*fcij/ros2
				#if(SB["normalize_by_ro"]): radial_term=radial_term/ros2
				#print(ros2,ros2.shape)

			if(SB['nn'].info['lsp_type']==20):
				#radial_term=np.exp(-((rij-s)/ros)**2.0)*np.exp(-(((rik-s)/ros))**2.0)*fcik*fcij 
				radial_term=np.exp(-((rij-0)/ros)**2.0)*np.exp(-(((rik-0)/ros))**2.0)*fcik*fcij 
			# print(ros.shape,radial_term.shape)
			# exit()
			#ANGULAR TERM
			first=True
			for m in range(0,max(lgs)+1):
				if(m==0): 
					lg_cos1=np.ones((cos_ijk1.shape[0],cos_ijk1.shape[1]))
				if(m==1): 
					lg_cos1_m1=lg_cos1;		lg_cos1=cos_ijk1;
				if(m in lgs): 
					if(first): 
						gis=(radial_term*(lg_cos1)).sum(axis=1); first=False
						#print(gis.shape)

						if(SB["normalize_by_ro"]==False): tmp_ros2=ros2

					else:
						if(SB["normalize_by_ro"]==False): 
							tmp_ros2=np.concatenate((tmp_ros2,ros2))

						gis=np.concatenate((gis,(radial_term*(lg_cos1)).sum(axis=1)))
						#print(gis.shape)

				if(m>=1): #define for next iteration of loop 
						tmp=lg_cos1
						lg_cos1=((2.0*m+1.0)*cos_ijk1*lg_cos1-m*lg_cos1_m1)/(m+1); lg_cos1_m1=tmp; 
			
			#exit()
			
			gis=np.arcsinh(gis)
			#print(gis.shape,(tmp_ros2.flatten()).shape); #exit()
			if(SB["normalize_by_ro"]==False): gis=(tmp_ros2.flatten())*gis; #exit()
			#nt(gis.shape,tmp_ros2.shape); exit()

			self.lsps.append(gis)

			if(SB['write_lsp']):
				str1=str(self.v)+" "+str(self.u)
				for i in gis:
					str1=str1+' %14.12f '%i
				str1=str1+" "+str(self.gid)
				writer.write_LSP(str1)