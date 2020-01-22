import 	numpy 		as 		np
import 	torch
import 	writer

# class DataSet:
# 	def __init__(self, SIDS):
# 		self.SIDS=SIDS

# 		#build essential arrays
# 		for i in 
# 			print(SIDS)

	# def :



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
		if(info['num_species']  != 1):						raise ValueError("NUM_SPECIES != 1 IN EURAL NETWORK FILE")
		if(len(info['nn_layers'])  != int(lines[7][0])):	raise ValueError("NUMBER OF LAYERS IN NEURAL NETWORK FILE IS INCORRECT")
		if(len(WB) != info['num_fit_param']):				raise ValueError("INCORRECT NUMBER OF FITTING PARAMETERS")
		if(int(lines[0][0]) not in [5,6,7]):				raise ValueError("REQUESTED POT_TYPE="+str(int(lines[0][0]))+" NOT AVAILABLE")

		#DEFINE OBJECT
		self.info = info					 
		self.WB = WB.tolist()    #long vector with all weights and biases
		#print(WB[0], WB[1],WB[-1]); exit()
		writer.write_NN(self,step=0)													#save a copy of the initial NN 

	#TAKES A LONG VECTOR W OF WEIGHTS AND BIAS AND RETURNS WEIGHT AND BIAS SUBMATRICES
	def extract_submatrices(self):
		submatrices=[]; K=0
		W=self.values
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
			#print(w.shape, b.shape)
		#print(self.values)

		return submatrices


class Structure:
	def __init__(self,lines,sid,params):
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
		self.E				= float(lines[-1]) #+ (self.n_atoms * e_shift)
		self.v				= self.V/self.N
		self.e				= self.E/self.N
		self.species		= params['species']  #TODO THIS NEEDS TO BE FIXED (GENERALIZE TO BINARY)

		print(self.a1.shape)
		exit()
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

	def compute_nbl_for_structure(self,params):

		pot_type=params['pot_type']
		Rc=params['cutoff_dist']

		#GET LENGTHS OF VECTORS
		a1_n = np.linalg.norm(self.a1)
		a2_n = np.linalg.norm(self.a2)
		a3_n = np.linalg.norm(self.a3)

		#FIND NUMBER OF TIMES TO REPEAT THE LATTICE (do +1 just to be safe)
		x_repeat = int(np.ceil(Rc / a1_n))  
		y_repeat = int(np.ceil(Rc / a2_n))  
		z_repeat = int(np.ceil(Rc / a3_n))  

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
		dtype = torch.FloatTensor

		#LOOP OVER ATOMS AND FIND NEIGHBORS WITHIN Rc
		ID=0
		for ri in self.positions:
				#array of distance vectors from atom ri 		
				rij	= periodic_structure - ri 		
				#magnetude of distance vector rij (axis=1-->accross row)					
				rij = np.linalg.norm(rij, axis = 1)					
				mask      = (rij > 1e-5) & (rij < Rc)
				neighbors = periodic_structure[mask]

				if(neighbors.shape[0]==0): 
					raise Exception("ATOM HAS NO NEIGHBORS",self.sid,self.comment)
				self.nbls.append(neighbors)
				ID += 1


		# print(self.nbls)
		# # print(self.sid,len(self.nbls)); exit()
		# if(self.nbls==[]): print(self.sid); exit()
		# exit()














	def compute_lsp_for_structure(self,params):
		rc=params['cutoff_dist']
		dc4=params['cutoff_range']**4.0
		#ros=np.array(params['lsp_ro_val']).reshape(len(params['lsp_ro_val']),1)
		ros=torch.tensor(params['lsp_ro_val']).view(len(params['lsp_ro_val']),1)

		ros2=(ros**2.0)
		lgs=params['lsp_lg_poly']
		s=params['lsp_sigma']
		dtype = torch.FloatTensor

		#LOOP OVER ATOMS AND FIND NEIGHBORS WITHIN Rc
		ID=0
		for ID in range(0,len(self.nbls)):


			##nbl=torch.tensor([[11,12,13],[21,22,23]])   #useful for testing what the following commands do
			#TEMP=
			ri=torch.tensor([1,0.0,0.], requires_grad=True).type(dtype) #.requires_grad_(True)
			nbl=torch.tensor(self.nbls[ID]).type(dtype)   #useful for testing what the following commands do

			#nbl=torch.from_numpy(self.nbls[ID]).requires_grad_(True).type(dtype)
			#temp=torch.tensor(self.positions[ID])
			#ri=torch.autograd.Variable(torch.from_numpy(self.positions[ID]), requires_grad=True).type(dtype) #.requires_grad_(True)

			#ri=temp.clone().detach().requires_grad_(True).type(dtype)
			#print(nbl[0][0],ri[0])
			nbl=nbl-ri
			print(nbl.shape,ri.shape)
			nbl.backward(ri)
			print(ri.grad)



			n=nbl.shape[0]  #num neigh
			Xij=(torch.cat([nbl]*n,1)).view(n*n,3)
			Xik=nbl.repeat((n,1))

			rij=((torch.sum(Xij**2.0,1))**0.5).view(1,n*n);		#	print(rij)		
			rik=((torch.sum(Xik**2.0,1))**0.5).view(1,n*n);			
			cos_ijk=(torch.sum(Xij*Xik,1).view(1,n*n))/rij/rik;		

			#RADIAL TERM
			fcij=(rij-rc)**4.0;  fcij=fcij/(dc4+fcij)
			# mask      =  (rij < rc).astype(np.int);		fcij=fcij*mask		

			fcik=(rik-rc)**4.0;  fcik=fcik/(dc4+fcik)
			# mask      =  (rik < rc).astype(np.int);		fcik=fcik*mask
			#term is the same for all LG poly (each row is a given ro)
			radial_term=torch.exp(-((rij-ros)/s)**2.0)*torch.exp(-((rik-ros)/s)**2.0)*fcik*fcij/ros2
			#print(radial_term.shape,ros.shape); 	exit()
			# print(ros); print(rij); print(rij-ros); print((rij-ros)*cos_ijk);
			# print(cos_ijk); print(radial_term)

			#ANGULAR TERM
			first=True
			for m in range(0,max(lgs)+1):
				if(m==0): lg_cos=torch.ones((cos_ijk.shape[0],cos_ijk.shape[1]))
				if(m==1): lg_cos_m1=lg_cos;		lg_cos=cos_ijk;
				if(m in lgs): 
					#print(lg_cos)
					if(first): 
						gis=torch.sum((radial_term*lg_cos),1); first=False
					else:
						gis=torch.cat((gis,torch.sum((radial_term*lg_cos),1)))
				if(m>=1): #define for next iteration of loop 
						tmp=lg_cos
						lg_cos=((2.0*m+1.0)*cos_ijk*lg_cos-m*lg_cos_m1)/(m+1); lg_cos_m1=tmp;



			# x = torch.tensor([1.5,2.1,4.0], requires_grad = True)
			# z = torch.sin(x**3.0)
			# z.backward(x)
			# print(x.grad/x) 
			# print(torch.cos(x**3.0)*3.0*x**2.0)
			# exit()
			# exit()
			# gis=gis.view(40,1)
			print(gis[0])
			print(ri[0])
			dg_dx=gis[0].backward(ri[0])
			print(dg_dx)
			exit()

			gis=torch.asinh(gis)
			self.lsps.append(gis)
			if(params['write_lsp']):
				str1=''
				for i in gis:
					str1=str1+' %14.12f '%i
				writer.write_LSP(str1)




		# print(centers)

		# # i, j = np.nested_iters(a, [[1], [0, 2]], flags=["multi_index"])

		# # print(itertools.product(range(3),range(3),range(3)))
		# centers=[]
		# for (i,j,k) in itertools.product(range(-x_repeat,x_repeat+1),range(-y_repeat,y_repeat+1),range(-z_repeat,z_repeat+1)):
		# 	centers.append(self.a1*i + self.a2*j + self.a3*k)
		# 	#print(i,j,k); # N=N+1

		# print(len(centers),N)
		# exit()
		# # print(x_repeat)
		# center_x=np.array(range(-x_repeat, x_repeat+1)); #center_x=center_x.reshape(1,len(center_x))
		# center_y=np.array(range(-y_repeat, y_repeat+1)); #center_y=center_y.reshape(len(center_y),1)
		# print(np.meshgrid(center_x,center_y))

		# exit()
		# # print(center_x)

		# # print(center_y)
		# # print(center_x+center_y)
		# indices=np.ndindex((x_repeat+1,y_repeat+1,z_repeat+1))
		# # print(centers.item())

		# for index  in indices:
		# 	print(index[0]*self.a1+index[1]*self.a2+index[2]*self.a3) 

		# 	print(index[0]*self.a1+index[1]*self.a2+index[2]*self.a3) 
		# exit()



		# 		# start = time.time()
		# first=True #self.positions.shape
		# for ri in self.positions:
		# 	if(first):
		# 		periodic_structure=ri+centers; first=False
		# 	else:
		# 		periodic_structure=np.concatenate((periodic_structure,ri+centers))
		# periodic_structure1=(centers+self.positions[:,None,:]).reshape(-1,self.positions.shape[1])  #add each row of positions to each row of 
		# print(np.array_equal(np.sort(periodic_structure1.flat), np.sort(periodic_structure1.flat)))


		# #GET CENTER LOCATIONS FOR PERIODIC IMAGES (-x_repeat*a1 to x_repeat*a1). 
		# first=True
		# for i in range(-x_repeat, x_repeat + 1):
		# 	for j in range(-y_repeat, y_repeat + 1):
		# 		for k in range(-z_repeat, z_repeat + 1):
		# 			center=torch.tensor([[float(i),float(j),float(k)]])
		# 			if(i==0 and j==0 and k==0): center.requires_grad = True

		# 			if(first): 
		# 				centers=center; first=False
		# 			else: 
		# 				centers=torch.cat((centers,center))

		# print(centers)
		# #centers=torch.tensor(centers); #lat_vec=np.array([self.a1,self.a2,self.a3])
		# exit()
		# centers=centers[:,0].reshape(len(centers),1)*self.a1+centers[:,1].reshape(len(centers),1)*self.a2+centers[:,2].reshape(len(centers),1)*self.a3

		# periodic_structure=(centers+self.positions[:,None,:]).reshape(-1,self.positions.shape[1])  #add each row of positions to each row of 

		# #for ri in self.positions:
