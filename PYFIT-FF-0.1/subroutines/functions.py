#import as little code as needed
from numpy import array, random,transpose 
from  os import  path 
from input_file import *  
from time import time
import numpy as np
from write import *    

#takes a long vector W of weights and bias and returns weight and bias submatrices
def matrix_extract(W,desc):

	len_desc=len(desc) #; print x,p,p.shape

	K=0; Wb=[] #k0=0; 
	for i in range(0,len_desc-1):
#		print "------LAYER",i,"to",i+1,"-------"
#		print "   SIZE:", desc[i],"x",desc[i+1] 

		#FORM RELEVANT SUB MATRIX FOR LAYER-N
		Nrow=desc[i+1]; Ncol=desc[i] #+1
		w=array(W[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
		K=K+Nrow*Ncol; #print i,k0,K
		Nrow=desc[i+1]; Ncol=1; #+1
		b=transpose(array([W[K:K+Nrow*Ncol]])) #unpack/ W 
		K=K+Nrow*Ncol; #print i,k0,K
		Wb.append(w); Wb.append(b)
		#print(w.shape, b.shape)
	return Wb


#takes an array of np.array submatrices and returns a long vector W of weights and bias  
def matrix_combine(Wb):
	#print(type(Wb[0]))
	W=[]
	for l  in range(0,len(Wb)):
		#print(Wb[l].shape,len(Wb[l][0]),len(Wb[l]))
		for j in range(0,len(Wb[l][0])): #len(w1[0])=number of columns
			for i in range(0,len(Wb[l])): #build down the each row then accros
				W.append(Wb[l][i][j])
	#print(len(W))
	return W


