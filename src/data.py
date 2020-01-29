
import	numpy	as	np
import  torch
import	writer
dtype		=	torch.FloatTensor
if(torch.cuda.is_available()): dtype = torch.cuda.FloatTensor

class Dataset:
	def __init__(self,name):
		self.name=name
		self.structures={};  #structure[SID]  --> structure_objects
		self.group_sids={}   #group_sids[GID] --> list of SID
		self.Na=0;
		self.Ns=0;


	def sort_group_sids(self): 
		for i in self.group_sids.keys():
			self.group_sids[i]=[item[1] for item in sorted(self.group_sids[i])]	 


	def build_arrays(self,SB): 

		#ALL MODELS NEED THE FOLLOWING 
		u1=[]; v1=[]; N1=[]; self.SIDS1=[]; self.GIDS1=[]	#DFT STUFF
		self.R1=torch.zeros(self.Ns,self.Na).type(dtype); j=0; k=0 # REDUCTION MATRIX: Ns X Na 
		for structure in self.structures.values():
				u1.append(structure.u)
				v1.append(structure.v)
				N1.append(structure.N)
				self.SIDS1.append(structure.sid)
				self.GIDS1.append(structure.gid)

				for i in range(0,structure.N):
				 	self.R1[j][k]=1
				 	k=k+1
				j=j+1

		self.v1=torch.tensor(np.transpose([v1])).type(dtype);
		self.u1=torch.tensor(np.transpose([u1])).type(dtype);
		self.N1=torch.tensor(np.transpose([N1])).type(dtype);

		if(SB['pot_type'] == "NN"):
			Gis=[] 
			for structure in self.structures.values():
				for Gi in structure.lsps:	Gis.append(Gi)
			self.Gis=torch.tensor(Gis).type(dtype);
			self.M1=torch.tensor([np.ones(len(self.Gis))]).type(dtype)


	#APPLY THE MODEL TO DATASET
	def evaluate_model(self,SB):
		#set_name=dateset object name
		if(SB['pot_type']=='NN'):
			#nn_out=self.NN_eval(SB['nn'])
			nn_out=SB['nn'].NN_eval(self)

			self.u2=(self.R1).mm(nn_out)/self.N1




	def report(self,SB,t):
		if(self.Ns!=0):
			self.evaluate_model(SB)
			writer.write_E_vs_V(self,t)
			RMSE=(((self.u1-self.u2)**2.0).sum()/self.Ns)**0.5 
			MAE=(((self.u1-self.u2)**2.0)**0.5).sum()/self.Ns
			MED_AE=torch.median((((self.u1-self.u2)**2.0)**0.5))
			STD_AE=torch.std(((self.u1-self.u2)**2.0)**0.5) 
			MAX_AE=torch.max(((self.u1-self.u2)**2.0)**0.5) 
			RMS_DU=0.5*((self.u1.view(self.u1.shape[0],1)-self.u1.view(1,self.u1.shape[0])) \
				-(self.u2.view(self.u2.shape[0],1)-self.u2.view(1,self.u2.shape[0]))) 
			RMS_DU=(torch.mean(RMS_DU**2.0)**0.5)
			writer.write_stats(self.name,t,RMSE,MAE,MED_AE,STD_AE,MAX_AE,RMS_DU)

			


	def compute_objective(self,SB):
		self.evaluate_model(SB)

		#OBJECTIVE TERM-1 (RMSE OR MAE)
		err=1000.0*(self.u1-self.u2) #convert to meV
		TMP=(torch.mean(err**2.0)**0.5)
		RMSE=TMP.item() #NOT USED IN OBJ 
		#err=torch.exp(-(self.u1+4.63)/1.0)*err
		if(RMSE<SB['rmse_xtanhx']): #RMAE>1 and RMSE<1
			OBE1=SB['lambda_E1']*(torch.mean(err*torch.tanh(err))) #**0.5
			#OBE1=SB['lambda_E1']*(torch.mean(torch.tanh(err)*torch.tanh(err))) #**0.5

		else:
			OBE1=SB['lambda_E1']*TMP

		#OBJECTIVE TERM-2 (DIFF)
		OB_DU=torch.tensor(0.0)
		if(SB['lambda_dU']>0): 
			DIFF1=1000.0*0.5*((self.u1.view(self.u1.shape[0],1)-self.u1.view(1,self.u1.shape[0])) \
			-(self.u2.view(self.u2.shape[0],1)-self.u2.view(1,self.u2.shape[0]))) 
			if(RMSE<SB['rmse_xtanhx']): #RMAE>1 and RMSE<1
				OB_DU=SB['lambda_dU']*(torch.mean(DIFF1*torch.tanh(DIFF1))) #**0.5
				#OB_DU=SB['lambda_dU']*(torch.mean(torch.tanh(DIFF1)*torch.tanh(DIFF1))) #**0.5
			else:
				OB_DU=SB['lambda_dU']*(torch.mean(DIFF1**2.0)**0.5)

		#OBJECTIVE TERMS 3 AND 4:  L1 AND L2 REG ON NN FITTING PARAM
		OBL1=0.0; OBL2=0.0
		if(SB['pot_type']=='NN'):
			#L1 REGULARIZATION
			S=0
			for i in range(0,len(SB['nn'].submatrices)): S=S+((SB['nn'].submatrices[i]**2.0)**0.5).sum()
			OBL1=SB['lambda_l1']*S

			#L2 REGULARIZATION 
			S=0
			for i in range(0,len(SB['nn'].submatrices)): S=S+(SB['nn'].submatrices[i]**2).sum()
			#LAMBDA_L2_REG=1*(-np.tanh(6*(t-25)/100)+1)*0.5+0.00001
			OBL2=SB['lambda_l2']*S

	
		return [RMSE,OBE1,OB_DU,OBL1,OBL2]









class Structure:
	def __init__(self,lines,sid,SB):
		#print(lines)
		SF=float(lines[1])	#SCALING FACTOR
		self.sid		= sid
		self.gid		= str(lines[0])

		self.scale_factor	= SF
		self.a1			= SF*(np.array(lines[2]).astype(np.float))
		self.a2			= SF*(np.array(lines[3]).astype(np.float))
		self.a3			= SF*(np.array(lines[4]).astype(np.float))
		self.V			= np.absolute(np.dot(self.a1,np.cross(self.a2,self.a3)))
		self.N      		= int(lines[5])
		self.U			= float(lines[-1])+self.N*SB['u_shift'] #+ (self.n_atoms * e_shift)
		self.v			= self.V/self.N
		self.u			= self.U/self.N
		self.species		= SB['species']  #TODO THIS NEEDS TO BE FIXED (GENERALIZE TO BINARY)

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
			cos_ijk=((Xij*Xik).sum(axis=1).reshape(1,n*n))/rij/rik;		

			#RADIAL TERM
			fcij=(rij-rc)**4.0;  fcij=fcij/(dc4+fcij)
			#mask      =  (rij < rc).astype(np.int);		fcij=fcij*mask		

			fcik=(rik-rc)**4.0;  fcik=fcik/(dc4+fcik)
			#mask      =  (rik < rc).astype(np.int);		fcik=fcik*mask


			#term is the same for all LG poly (each row is a given ro)
			radial_term=np.exp(-((rij-ros)/s)**2.0)*np.exp(-((rik-ros)/s)**2.0)*fcik*fcij/ros2

			#print(radial_term.shape,rij.shape,ros.shape); exit()


			#ANGULAR TERM
			first=True
			for m in range(0,max(lgs)+1):
				if(m==0): lg_cos=np.ones((cos_ijk.shape[0],cos_ijk.shape[1]))
				if(m==1): lg_cos_m1=lg_cos;		lg_cos=cos_ijk;
				if(m in lgs): 
					if(first): 
						gis=(radial_term*lg_cos).sum(axis=1); first=False
					else:
						gis=np.concatenate((gis,(radial_term*lg_cos).sum(axis=1)))
					#print(gis.shape)

				if(m>=1): #define for next iteration of loop 
						tmp=lg_cos
						lg_cos=((2.0*m+1.0)*cos_ijk*lg_cos-m*lg_cos_m1)/(m+1); lg_cos_m1=tmp;


			#print(radial_term.shape)
			#print(gis.shape)
			#exit()

			gis=np.arcsinh(gis)
			self.lsps.append(gis)

			if(SB['write_lsp']):
				str1=''
				for i in gis:
					str1=str1+' %14.12f '%i
				writer.write_LSP(str1)