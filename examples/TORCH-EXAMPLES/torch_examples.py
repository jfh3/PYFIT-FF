import  numpy	as	np
import 	torch 


#-----------------------------------------------------
#PYTORCH BASIC OPERATIONS 
#-----------------------------------------------------

A=torch.tensor([[11., 12, 13], [21,22,23]]);	#DEFINE TORCH TENSOR	
B=np.array([[31., 32, 33]]);			
B=torch.tensor(B)				#DEFINE TORCH TENSOR FROM NUMPY ARRAY

print(A,B)
print(A.shape,B.shape)

print(A+1.5)

print(A+B)
print(torch.sigmoid(A))
print(A*B)
print(A**2.0)

print(torch.transpose(A,0,1))
print(torch.t(A))


exit()
#print(np.tile(nbl,n).reshape(n*n,3)
#Xik=np.tile(nbl,(n,1)).reshape(n*n,3) #.reshape(3*n,1) #.repeat(n,axis=0).reshape(n*n,3)

rij=(((Xij**2.0).sum(axis=1))**0.5).reshape(1,n*n); #sum accros row		
rik=(((Xik**2.0).sum(axis=1))**0.5).reshape(1,n*n);	


exit()



#------------------------------------------------------------------
#OPERATIONS USED BY PYTORCH:
#------------------------------------------------------------------




print("#---------------REDUCTION METHOD-1---------------")
#MASKED SCATTER (3-6X MORE EFFICENT THAN PREVIOUS R MATRIX METHOD)
	#GIVEN A VECTOR OF ATOMIC ENERGIES PREFORM A REDUCTION TO SUM OVER
	#RELEVANT INDICES TO OUTPUT VECTOR OF STRUCTURAL ENERGIES 

Na=4	#total number of atoms in data set
Ns=3	#total number of structures in data set

#EXAMPLE-1: data set with 4 atoms, 3 structures with Natoms=(2,1,1) respectivly
#max(N_atom)=2 in this case 

#vector with "energies" Natom X 1 
u2 = torch.tensor([[1.2],[2.8],[3.0],[4.5]]).type(torch.FloatTensor) 		

#precomputed mask to select each structure's atp,s
	#mask=Nstructure X max(N_atom) 
mask = torch.tensor([[True],[True],[True],[False],[True],[False]] ) 
start=time()	
A=torch.zeros(6, 1).masked_scatter_(mask, u2)
print(A)
print(A.view(3,2)) #row=structure  #columns=relevant atoms
print(torch.sum(A.view(3,2),1)) #sum over row
print(['OPERTAION TIME (SEC):',time()-start])
# exit()




#------------------------------------------------------------------
#ADDITIONAL OPERATIONS FOR REFERENCE:
#------------------------------------------------------------------

#MASKED SCATTER PREFORM REDUCTION  
# u2=torch.tensor([[1,2,3,4,5,6,7,8]]).type(torch.FloatTensor) 
# mask = torch.tensor([[True],[True],[False],[False],[True],[True],[False],[False],[True],[True],[True],[True]]) 
# A=torch.zeros(12, 1).masked_scatter_(mask, u2)
# print(A)
# print(A.view(3,4)) #row=structure  #columns=relevant atoms
# print(torch.sum(A.view(3,4),1)) #sum over row

#SCATTER
#scatter_(dim, index, src) → Tensor
# Writes all values from the tensor src into self at the indices specified in the index tensor. 
# For each value in src, its output index is specified by its index in src for dimension != 
# dim and by the corresponding value in index for dimension = dim.

# x = torch.tensor([[1],[2],[3]]).type(torch.FloatTensor) #torch.ones(3, 1)
# ind=torch.tensor([[0],[1], [2]])
# print(x,x.shape, type(x))
# A=torch.zeros(4, 1).scatter_(0,ind , x)
# print(A)
# B=A.scatter_(0,ind , x)
# print(B)
# exit()

#GATHER
# #torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor
# #Gathers values along an axis specified by dim.
# t = torch.tensor([[1,2],[3,4]])
# print(t)
# print(torch.gather(t, 1, torch.tensor([[0,1],[0,1]])))
# print(torch.gather(t, 1, torch.tensor([[1,1],[0,1]])))
# # A = torch.tensor([[0, 1], [2, 3], [5, 6]]).float()
# # B = torch.tensor([[1], [1], [0]]).long()
# # print(torch.gather(A, 1, B))
# exit()

# #MASKED SCATTER
# x = torch.tensor([[1],[2],[3]]).type(torch.FloatTensor) #torch.ones(3, 1)
# print(x,x.shape, type(x))
# mask = torch.tensor([[True],[True],[False],[True]] ) #.ge(0.5)
# A=torch.zeros(4, 1).masked_scatter_(mask, x)
# print(A)

# x = torch.tensor([[1],[2],[3],[4]]).type(torch.FloatTensor) #torch.ones(3, 1)
# print(x,x.shape, type(x))
# mask = torch.tensor([[True],[True],[True],[False],[True],[False]] ) #.ge(0.5)
# A=torch.zeros(6, 1).masked_scatter_(mask, x)
# print(A)
# exit()

# #MASKED SELECT
# x = torch.randn(4, 1)
# y = torch.randn(3, 2)
# mask = torch.tensor([[True,True],[True,True],[True,True]] ) #.ge(0.5)
# print(x)
# print(mask)
# print(torch.masked_select(x, mask))
# exit()

# #USE CUMSUM TO PREFORM REDUCTION (BAD METHOD, DONT USE!! ,LARGE ERROR ACCUMULATION FOR LONG VECTORS)
# a=torch.tensor([[1,2,3,4,5,6,7,8]])
# b=a.cumsum(1) #cumulative sum over row
# c=b.gather(1, torch.tensor([[1,3,7]])) #select relevant terms
# d=torch.cat( (torch.tensor([[0]]), b.gather(1, torch.tensor([[1,3]]))),1) #select relevant terms
# print(c,d,c-d)
