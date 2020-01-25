import 	numpy 		as 		np
import 	writer
import  torch

dtype		=	torch.FloatTensor
if(torch.cuda.is_available()): dtype = torch.cuda.FloatTensor

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

		if(info['randomize_nn']==True or SB['re_randomize']): 
			writer.log(["	 RANDOMIZING NN MIN/MAX	=",info['max_rand_wb']])
			WB	=	np.random.uniform(-info['max_rand_wb'],info['max_rand_wb'],nfit)
			# WB	=	np.random.normal(0.0,info['max_rand_wb'],nfit)
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
		self.submatrices=self.extract_submatrices(WB)
		writer.write_NN(self,step=0)													#save a copy of the initial NN 

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
			for i in self.submatrices:	i.requires_grad = True

	# #GIVEN AN INPUT MATRIX EVALUATE THE NN
	def NN_eval(self,x):
		# print(type(self.submatrices[0]))
		#M1=torch.tensor([np.ones(len(x))]).type(dtype)
		out=(x.Gis).mm(torch.t(self.submatrices[0]))+torch.t((self.submatrices[1]).mm(x.M1))
		for i in range(2,int(len(self.submatrices)/2+1)):
			j=2*(i-1)
			out=torch.sigmoid(out)	
			if(self.info['activation']==1):
				out=out-0.5	
			out=out.mm(torch.t(self.submatrices[j]))+torch.t((self.submatrices[j+1]).mm(x.M1))
		return out 





# def compute(set_name,SB):
# 	#set_name=dateset object name
# 	print(SB[set_name])
# 	if( SB['pot_type']=='NN'): 
# 		nn_out=SB['pot'].NN_eval(SB[set_name].Gis)
# 		u2=(SB[set_name].R1).mm(nn_out)/SB[set_name].N1
	
# 	#print(u2-SB[set_name].u1)

# 	RMSE=(((SB[set_name].u1-u2)**2.0).sum()/len(u2))**0.5	
# 	print(RMSE)

# 	# def construct_matrices(SB):

# # 	if(SB['pot_type'] != "NN"): raise ValueError("REQUESTED MODEL NOT CODED YET");




# # 		U2=R1.mm(nn_out)
# # 		u2=U2/N1
# # 		u1=U1/N1

# # 		print(Gis.shape,nn_out.shape,R1.shape,U1.shape,U2.shape)



