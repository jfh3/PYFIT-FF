import 	numpy	as	np
import 	writer
import  torch

class NN:
	#USED FOR ANN,PINN, PNN
	def __init__(self, lines, SB):
		
		info={}	
		info['lsp_type']		=	 int(lines[0][0])
		info['pot_type']		=	 str(SB['pot_type'])
		info['lsp_shift']		=	 float(lines[0][1])
		info['activation']		=	 int(lines[0][2])
		info['num_species']		=	 int(lines[1][0])
		info['species']			=	 str(lines[2][0])
		info['atomic_weight']		=	 float(lines[2][1])
		info['randomize_nn']		=	 bool(int(lines[3][0]))
		info['max_rand_wb']		=	 float(lines[3][1])
		info['cutoff_dist']		=	 float(lines[3][2])
		info['cutoff_range']		=	 float(lines[3][3])
		info['lsp_sigma']		=	 float(lines[3][4])
		info['N_lg_poly']		=	 int(lines[4][0])	
		#map converts str list to int list 
		info['lsp_lg_poly']		=	 list(map(int,lines[4][1:]))	
		info['N_ro_val']		=	 int(lines[5][0])	 
		info['lsp_ro_val']		=	 list(map(float,lines[5][1:]))	
		#map converts str list to float list
		info['ibaseline']		=	 bool(int(lines[6][0]))
		info['bop_param']		=	 list(map(float,lines[6][1:]))  
		info['nn_layers']		=	 list(map(int,lines[7][1:]))  
		info['cnst_final_bias']		=	 SB['cnst_final_bias'] 
		info['final_bias']		=	 SB['final_bias'] 
		info['start_fresh']		=	 SB['start_fresh'] 
		info['constrain_WB']		=	 SB['constrain_WB'] 

		#DETERMINE NUMBER OF FITITNG PARAM AND RANDOMIZE IF NEEDED
		nfit=0; layers=info['nn_layers']
		for i in range(1,len(layers)):  nfit=nfit+layers[i-1]*layers[i]+layers[i]
		info['num_fit_param']	 =	nfit

		self.info = info

		self.normalize_by_ro=SB["normalize_by_ro"]

		self.dtype=torch.FloatTensor
		if(SB['use_cuda']): self.dtype=torch.cuda.FloatTensor

		if(info['randomize_nn'] or SB['re_randomize']): 
			writer.log(["	 RANDOMIZING NN MIN/MAX	=",info['max_rand_wb']])
			self.randomize(); 		
		else:
			#always do LR ramp up when re-starting
			SB['ramp_LR']	=	True 
			WB		=	np.array(lines[8:]).astype(np.float)[:,0]
			self.submatrices=self.extract_submatrices(WB)
			if(len(WB) != info['num_fit_param']):				
				raise ValueError("INCORRECT NUMBER OF FITTING PARAMETERS")

		#SOME ERROR CHECKS 
		if(info['num_species']  != 1):						
			raise ValueError("NUM_SPECIES != 1 IN EURAL NETWORK FILE")
		if(len(info['nn_layers'])  != int(lines[7][0])):	
			raise ValueError("NUMBER OF LAYERS IN NEURAL NETWORK FILE IS INCORRECT")

		if(int(lines[0][0]) not in [5,6,7,20]):				
			raise ValueError("REQUESTED POT_TYPE="+str(int(lines[0][0]))+" NOT AVAILABLE")
		if( info['pot_type']=='PINN_BOP' and info['nn_layers'][-1]!=8): 
			raise ValueError("ERROR: NN OUTPUT DIMENSION INCORRECT")
		if( info['pot_type']=='NN' and info['nn_layers'][-1]!=1): 
			raise ValueError("ERROR: NN OUTPUT DIMENSION INCORRECT")
		if( info['N_ro_val']  != len(info['lsp_ro_val'])): 
			raise ValueError("ERROR: N_ro_val != len(ro)")
		if( info['N_lg_poly']  != len(info['lsp_lg_poly'])): 
			raise ValueError("ERROR: N_lg_poly != len(lsp_lg_poly)")
		if( info['nn_layers'][0] != len(info['lsp_ro_val'])*len(info['lsp_lg_poly'])): 
			raise ValueError("ERROR: NN INPUT DIMENSION INCORRECT FOR Gi CHOICE")
		#self.add_neurons()

	def randomize(self):
		WB	=	np.random.uniform(-self.info['max_rand_wb'],self.info['max_rand_wb'],self.info['num_fit_param'])
		self.submatrices=self.extract_submatrices(WB)
		self.set_grad()

	#ADD NEURON TO FIRST LAYER  
	def add_neurons(self):
		#add N neurons to each hidden layer 

		if( self.info['activation'] != 1): 
			raise ValueError("ERROR: CAN ONLY ADD NEURONS TO SHIFTD SIGMOID FUNCTION")

		#START FRESH EVERY TIME
		start_fresh=self.info['start_fresh'] 
		if(start_fresh):
			self.randomize(); 		
			max_rand_wb=self.info['max_rand_wb']
		else:
			max_rand_wb=1.0

		self.unset_grad()

		new_nfit=0
		N_neuron_2_add=2
		writer.log("ADDING "+str(N_neuron_2_add)+" NEURONS TO EACH LAYER")
		writer.log(["	original num_fit_param	=",self.info['num_fit_param']])

		for layer_add in range(1,len(self.info['nn_layers'])-1):
			for neurons in range(0,N_neuron_2_add):
				for i in range(0,len(self.submatrices)):
					layer=2*(i-1)

					if(layer_add==(i+2.0)/2):
						#ADD ROW (WEIGHT MATRIX)
						shp2=self.submatrices[i].shape[1]
						TMP=max_rand_wb*torch.empty(1,shp2).uniform_(-1.0, 1.0)
						self.submatrices[i]=torch.cat((self.submatrices[i],TMP))

						#ADD BIAS 
						shp2=self.submatrices[i+1].shape[1]
						TMP=max_rand_wb*torch.empty(1,shp2).uniform_(-1.0, 1.0)
						self.submatrices[i+1]=torch.cat((self.submatrices[i+1],TMP))

						# #ADD COL (WEIGHT MATRIX)
						shp1=self.submatrices[i+2].shape[0]
						TMP=max_rand_wb*torch.empty(shp1,1).uniform_(-1.0, 1.0)
						self.submatrices[i+2]=torch.cat((self.submatrices[i+2],TMP),1)

						self.info['nn_layers'][layer_add]=self.info['nn_layers'][layer_add]+1

		#COUNT NFIT
		for i in range(0,len(self.submatrices)):
			new_nfit+=self.submatrices[i].shape[0]*self.submatrices[i].shape[1]

		self.info['num_fit_param']=new_nfit
		writer.log(["	new num_fit_param	=",new_nfit])

		self.set_grad()

	#TAKES AN ARRAY OF NP.ARRAY SUBMATRICES AND RETURNS A LONG VECTOR W OF WEIGHTS AND BIAS  
	def matrix_combine(self):
		W=[]

		if(self.normalize_by_ro==False):
			#ros1=np.array(SB['lsp_ro_val']).reshape(len(SB['lsp_ro_val']),1)

			tmp=self.info['lsp_ro_val'] #torch.tensor(self.info['lsp_ro_val'])
			for m in range(1,self.info['N_lg_poly']):
				tmp=tmp+self.info['lsp_ro_val'] #np.concatenate((tmp,torch.tensor(self.info['lsp_ro_val'])))
			#print(tmp)
			tmp=torch.tensor([tmp])
			#print(tmp.shape)

		for l in range(0,len(self.submatrices)):
	
			if(l==0 and self.normalize_by_ro==False):
				#print(self.submatrices[l].shape,tmp.shape) #,tmp.view(1,30),tmp.view(1,30).shape)
				#print(self.submatrices[l])
				TMP=self.submatrices[l]*(tmp**2) #tmp.view(1,30)*tmp.view(1,30)
				#print(self.submatrices[l])
				#print(TMP)
				#print(tmp)
				#exit()

			#print(self.submatrices[l])
			for j in range(0,len(self.submatrices[l][0])): #len(w1[0])=number of columns
				for i in range(0,len(self.submatrices[l])): #build down the each row then accros
					if(l==0 and self.normalize_by_ro==False):
						W.append(self.F(TMP[i][j]).item())
						#print("HERE"); exit()
					else:
						W.append(self.F(self.submatrices[l][i][j]).item())
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
				submatrices[i]=torch.tensor(submatrices[i]).type(self.dtype)
				#writer.log("	 matrix_shape		:	"+str(submatrices[i].shape))

		return submatrices

	def send_to_gpu(self):
		for i in range(0,len(self.submatrices)):	
			self.submatrices[i]=self.submatrices[i].cuda(); #print(i.is_cuda)

	def send_to_cpu(self):
		for i in range(0,len(self.submatrices)):	
			self.submatrices[i]=self.submatrices[i].cpu(); #print(i.is_cuda)

	def unset_grad(self):
		for i in range(0,len(self.submatrices)): self.submatrices[i].requires_grad = False

	def set_grad(self):
		if(self.info['pot_type']=='NN'):
			for i in range(0,len(self.submatrices)):	
				 self.submatrices[i].requires_grad = True
			if(self.info['cnst_final_bias']): 
				self.submatrices[-1].requires_grad=False
				self.submatrices[-1]=(self.info['final_bias']* \
				torch.ones(self.submatrices[-1].shape[0],self.submatrices[-1].shape[1])).type(self.dtype)

	#FUNCTION FOR SMOOTHLY CONSTRAINING NN WEIGHTS AND BIAS TERMS (SMOOTH MAX)
	def F(self,x):
		if(self.info['constrain_WB']>0):
			return  x*torch.exp(-(x*0.429/self.info['constrain_WB'])**2)
		else:
			return x

	# #GIVEN AN INPUT MATRIX EVALUATE THE NN
	def NN_eval(self,x):
		MAX1=torch.max(torch.max(self.F(self.submatrices[1])),torch.max(self.F(self.submatrices[0])))

		out=(x.Gis).mm(torch.t(self.F(self.submatrices[0])))+torch.t((self.F(self.submatrices[1])).mm(x.M1))

		for i in range(2,int(len(self.submatrices)/2+1)):
			j=2*(i-1)

			TMP=torch.max(torch.max(self.F(self.submatrices[j])),torch.max(self.F(self.submatrices[j+1])))
			MAX1=torch.max(TMP,MAX1)

			if(self.info['activation']==0 or self.info['activation']==1 ):  
				out=torch.sigmoid(out)	
				if(self.info['activation']==1):
					out=out-0.5	
				out=out.mm(torch.t(self.F(self.submatrices[j])))+torch.t((self.F(self.submatrices[j+1])).mm(x.M1))

		#self.maxwb=self.F(MAX1)
		self.maxwb=MAX1


		return out 





		# print(self.info['nn_layers'],self.info['num_fit_param'])
		# for i in range(0,len(self.submatrices)):
		# 	print(self.submatrices[i].shape)
