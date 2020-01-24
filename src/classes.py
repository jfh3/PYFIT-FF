import 	numpy 		as 		np
import 	writer
import  torch

dtype		=	torch.FloatTensor
if(torch.cuda.is_available()): dtype = torch.cuda.FloatTensor


class Neural_Network:
	def __init__(self, lines):
		#STORE NN HEADER LINES IN DICTIONARY
		
		info={}	
		if(int(lines[0][0])==5): pot_type='NN'
		if(int(lines[0][0])==6): pot_type='PINN'
		if(int(lines[0][0])==7): pot_type='BOP'
		
		info['lsp_type']		=	 int(lines[0][0])
		info['pot_type']		=	 str(pot_type)
		info['lsp_shift']		=	 float(lines[0][1])
		info['activation']		=	 int(lines[0][2])
		info['num_species']		=	 int(lines[1][0])
		info['species']			=	 str(lines[2][0])

		info['atomic_weight']	=	 float(lines[2][1])
		info['randomize_nn']	=	 bool(int(lines[3][0]))
		info['max_rand_wb']		=	 float(lines[3][1])
		info['cutoff_dist']		=	 float(lines[3][2])
		info['cutoff_range']	=	 float(lines[3][3])
		info['lsp_sigma']		=	 float(lines[3][4])
		info['lsp_lg_poly']		=	 list(map(int,lines[4][1:]))	#map converts str list to int list
		info['lsp_ro_val']		=	 list(map(float,lines[5][1:]))	#map converts str list to float list
		info['ibaseline']		=	 bool(int(lines[6][0]))
		info['bop_param']		=	 list(map(float,lines[6][1:]))  
		info['nn_layers']		=	 list(map(int,lines[7][1:]))  

		#DETERMINE NUMBER OF FITITNG PARAM AND RANDOMIZE IF NEEDED
		nfit=0; layers=info['nn_layers']
		for i in range(1,len(layers)):  nfit=nfit+layers[i-1]*layers[i]+layers[i]
		info['num_fit_param']	 =	nfit
		if(info['randomize_nn']==True): 
			WB	=	np.random.uniform(-nn_info['max_rand_wb'],nn_info['max_rand_wb'],nfit)
		else:
			WB	=	np.array(lines[8:]).astype(np.float)[:,0]

		#SOME ERROR CHECKS 
		if(info['num_species']  != 1):						
			raise ValueError("NUM_SPECIES != 1 IN EURAL NETWORK FILE")
		if(len(info['nn_layers'])  != int(lines[7][0])):	
			raise ValueError("NUMBER OF LAYERS IN NEURAL NETWORK FILE IS INCORRECT")
		if(len(WB) != info['num_fit_param']):				
			raise ValueError("INCORRECT NUMBER OF FITTING PARAMETERS")
		if(int(lines[0][0]) not in [5,6,7]):				
			raise ValueError("REQUESTED POT_TYPE="+str(int(lines[0][0]))+" NOT AVAILABLE")
		if( info['pot_type']=='PINN_BOP' and info['nn_layers'][-1]!=8): 
			raise ValueError("ERROR: NN OUTPUT DIMENSION INCORRECT")
		if( info['pot_type']=='NN' and info['nn_layers'][-1]!=1): 
			raise ValueError("ERROR: NN OUTPUT DIMENSION INCORRECT")

		#DEFINE OBJECT
		self.info = info					 
		#self.WB = WB.tolist()    #long vector with all weights and biases (only used for writing)
		self.submatrices=self.extract_submatrices(WB)
		writer.write_NN(self,step=0)													#save a copy of the initial NN 

	#TAKES AN ARRAY OF NP.ARRAY SUBMATRICES AND RETURNS A LONG VECTOR W OF WEIGHTS AND BIAS  
	def matrix_combine(self):
		W=[]
		TMP=self.submatrices
		for l in range(0,len(TMP)):
			for j in range(0,len(TMP[l][0])): #len(w1[0])=number of columns
				for i in range(0,len(TMP[l])): #build down the each row then accros
					W.append(TMP[l][i][j].item())
		TMP=0.0
		return W

	#TAKES A LONG VECTOR W OF WEIGHTS AND BIAS AND RETURNS WEIGHT AND BIAS SUBMATRICES
	def extract_submatrices(self,WB):
		submatrices=[]; K=0
		W=np.array(WB)
		nn_layers=self.info['nn_layers']
		for i in range(0,len(nn_layers)-1):
			#FORM RELEVANT SUB MATRIX FOR LAYER-N
			Nrow=nn_layers[i+1]; Ncol=nn_layers[i] #+1
			w=np.array(W[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
			K=K+Nrow*Ncol; #print i,k0,K
			Nrow=nn_layers[i+1]; Ncol=1; #+1
			b=np.transpose(np.array([W[K:K+Nrow*Ncol]])) #unpack/ W 
			K=K+Nrow*Ncol; #print i,k0,K
			submatrices.append(w); submatrices.append(b)

		#CONVERT TO TORCH TENSORS
		for i in range(0,len(submatrices)):
				submatrices[i]=torch.tensor(submatrices[i]).type(dtype)
				writer.log(" 	SUBMATRIX SHAPE		:	"+str(submatrices[i].shape))

		return submatrices

	def unset_grad(self):
		for i in self.submatrices:	i.requires_grad = False

	def set_grad(self):
		if(info['pot_type']=='NN'):
			for i in self.submatrices:	i.requires_grad = True

	#GIVEN AN INPUT MATRIX EVALUATE THE NN
	def ANN(self,x):
		# print(type(self.submatrices[0]))
		M1=torch.tensor([np.ones(len(x))]).type(dtype)
		out=x.mm(torch.t(self.submatrices[0]))+torch.t((self.submatrices[1]).mm(M1))
		for i in range(2,int(len(self.submatrices)/2+1)):
			j=2*(i-1)
			out=torch.sigmoid(out)	
			if(self.info['activation']==1):
				out=out-0.5	
			out=out.mm(torch.t(self.submatrices[j]))+torch.t((self.submatrices[j+1]).mm(M1))
		return out 


class Structure:
	def __init__(self,lines,sid,SB):
		#print(lines)
		SF=float(lines[1])	#SCALING FACTOR
		self.sid			= sid
		self.comment		= str(lines[0])
		self.scale_factor	= SF
		self.a1				= SF*(np.array(lines[2]).astype(np.float))
		self.a2				= SF*(np.array(lines[3]).astype(np.float))
		self.a3				= SF*(np.array(lines[4]).astype(np.float))
		self.V				= np.absolute(np.dot(self.a1,np.cross(self.a2,self.a3)))
		self.N      		= int(lines[5])
		self.U				= float(lines[-1])+self.N*SB['u_shift'] #+ (self.n_atoms * e_shift)
		self.v				= self.V/self.N
		self.u				= self.U/self.N
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
					raise Exception("ATOM HAS NO NEIGHBORS",self.sid,self.comment)

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
			#nbl=np.array([[0,3,0],[2,0,0]]) #np.array([[11,12,13],[21,22,23]])   #useful for testing what the following commands do

			n=nbl.shape[0]
			Xij=np.tile(nbl,n).reshape(n*n,3)
			Xik=np.tile(nbl,(n,1)).reshape(n*n,3) #.reshape(3*n,1) #.repeat(n,axis=0).reshape(n*n,3)

			rij=(((Xij**2.0).sum(axis=1))**0.5).reshape(1,n*n);		#	print(rij)		
			rik=(((Xik**2.0).sum(axis=1))**0.5).reshape(1,n*n);			
			cos_ijk=((Xij*Xik).sum(axis=1).reshape(1,n*n))/rij/rik;		

			#RADIAL TERM
			fcij=(rij-rc)**4.0;  fcij=fcij/(dc4+fcij)
			#mask      =  (rij < rc).astype(np.int);		fcij=fcij*mask		

			fcik=(rik-rc)**4.0;  fcik=fcik/(dc4+fcik)
			#mask      =  (rik < rc).astype(np.int);		fcik=fcik*mask

			#term is the same for all LG poly (each row is a given ro)
			radial_term=np.exp(-((rij-ros)/s)**2.0)*np.exp(-((rik-ros)/s)**2.0)*fcik*fcij/ros2

			#ANGULAR TERM
			first=True
			for m in range(0,max(lgs)+1):
				if(m==0): lg_cos=np.ones((cos_ijk.shape[0],cos_ijk.shape[1]))
				if(m==1): lg_cos_m1=lg_cos;		lg_cos=cos_ijk;
				if(m in lgs): 
					#print(lg_cos)
					if(first): 
						gis=(radial_term*lg_cos).sum(axis=1); first=False
					else:
						gis=np.concatenate((gis,(radial_term*lg_cos).sum(axis=1)))
				if(m>=1): #define for next iteration of loop 
						tmp=lg_cos
						lg_cos=((2.0*m+1.0)*cos_ijk*lg_cos-m*lg_cos_m1)/(m+1); lg_cos_m1=tmp;

			gis=np.arcsinh(gis)
			self.lsps.append(gis)

			if(SB['write_lsp']):
				str1=''
				for i in gis:
					str1=str1+' %14.12f '%i
				writer.write_LSP(str1)
