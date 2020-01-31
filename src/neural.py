import 	numpy 		as 		np
import 	writer
import  torch

dtype		=	torch.FloatTensor
if(torch.cuda.is_available()): dtype = torch.cuda.FloatTensor
#S
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
		info['lsp_lg_poly']		=	 list(map(int,lines[4][1:]))	#map converts str list to int list
		info['lsp_ro_val']		=	 list(map(float,lines[5][1:]))	#map converts str list to float list
		info['ibaseline']		=	 bool(int(lines[6][0]))
		info['bop_param']		=	 list(map(float,lines[6][1:]))  
		info['nn_layers']		=	 list(map(int,lines[7][1:]))  

		info['cnst_final_bias']		=	 SB['cnst_final_bias'] 
		info['final_bias']		=	 SB['final_bias'] 


		#DETERMINE NUMBER OF FITITNG PARAM AND RANDOMIZE IF NEEDED
		nfit=0; layers=info['nn_layers']
		for i in range(1,len(layers)):  nfit=nfit+layers[i-1]*layers[i]+layers[i]
		info['num_fit_param']	 =	nfit

		if(info['randomize_nn']==True or SB['re_randomize']): 
			writer.log(["	 RANDOMIZING NN MIN/MAX	=",info['max_rand_wb']])
			WB	=	np.random.uniform(-info['max_rand_wb'],info['max_rand_wb'],nfit)
			#WB	=	np.random.normal(0.0,info['max_rand_wb'],nfit)
		else:
			#always do LR ramp up when re-starting
			SB['ramp_LR']	=	True 
			WB		=	np.array(lines[8:]).astype(np.float)[:,0]

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
		self.submatrices=self.extract_submatrices(WB)
		#writer.write_NN(self,step=0)													#save a copy of the initial NN 

	#TAKES AN ARRAY OF NP.ARRAY SUBMATRICES AND RETURNS A LONG VECTOR W OF WEIGHTS AND BIAS  
	def matrix_combine(self):
		W=[]
		for l in range(0,len(self.submatrices)):
			for j in range(0,len(self.submatrices[l][0])): #len(w1[0])=number of columns
				for i in range(0,len(self.submatrices[l])): #build down the each row then accros
					W.append(self.submatrices[l][i][j].item())
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
				writer.log("	 matrix_shape		:	"+str(submatrices[i].shape))

		return submatrices

	def unset_grad(self):
		for i in self.submatrices:	i.requires_grad = False

	def set_grad(self):
		if(self.info['pot_type']=='NN'):
			for i in self.submatrices:	
				i.requires_grad = True
			if(self.info['cnst_final_bias']): 
				#print(self.submatrices[-1]); #exit()
				self.submatrices[-1].requires_grad=False
				self.submatrices[-1]=self.info['final_bias']* \
				torch.ones(self.submatrices[-1].shape[0],self.submatrices[-1].shape[1])
				#print(self.submatrices[-1]); exit()

	# #GIVEN AN INPUT MATRIX EVALUATE THE NN
	def NN_eval(self,x):
		#ACTS ON A DATASET OBJECT
		out=(x.Gis).mm(torch.t(self.submatrices[0]))+torch.t((self.submatrices[1]).mm(x.M1))
		for i in range(2,int(len(self.submatrices)/2+1)):
			j=2*(i-1)

			if(self.info['activation']==0 or self.info['activation']==1 ):  
				out=torch.sigmoid(out)	
				if(self.info['activation']==1):
					out=out-0.5	
				out=out.mm(torch.t(self.submatrices[j]))+torch.t((self.submatrices[j+1]).mm(x.M1))

			if(self.info['activation']==10): #ALT ACTIVATION SCHEME
				#ADD ERROR FOR MULTLATER
				out=(1.0-torch.exp(-out))**2.0-1.0
	
				out=out.mm(torch.t(self.submatrices[j]))+torch.t((self.submatrices[j+1]).mm(x.M1))


		#out=(1.0-torch.exp(-out))**2.0	
		#xexp2=True; max_xexp2=10.0
		#if(xexp2):  out=max_xexp2*out*torch.exp(-1.0*out**2.0)/.4288	
			
	
		return out 


