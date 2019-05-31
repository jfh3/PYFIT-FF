import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim

# This class handles the process of loading information from the raw
# set of neural network weights and biases into a proper PyTorch neural
# network class. This is also the object that is actually called to 
# evaluate the network. It also contains functionality for dumping the
# values back out into a format that can be written into a file again.
class TorchNet(nn.Module):
	def __init__(self, network_data, reduction_matrix):
		super(TorchNet, self).__init__()

		# Here we need to instantiate and instance of a linear
		# transformation for each real layer of the network and
		# populate its data members with the weights and biases
		# that we want.

		# We have to use the ParameterList class here because the 
		# regular Python list isn't recognized by PyTorch and 
		# subsequent calls to TorchNet.parameters() will not work.
		self.layers           = []
		self.params           = nn.ParameterList()
		self.activation_mode  = network_data.activation_function
		self.reduction_matrix = reduction_matrix

		# Create a set of linear transforms.
		for idx in range(len(network_data.layer_sizes)):
			if idx != 0:
				prev_layer_size = network_data.layer_sizes[idx - 1]
				curr_layer_size = network_data.layer_sizes[idx]
				layer           = nn.Linear(prev_layer_size, curr_layer_size)
				current_layer     = network_data.layers[idx - 1]
				with torch.no_grad():
					layer.weight.copy_(torch.tensor([node[0] for node in current_layer]))
					layer.bias.copy_(torch.tensor([node[1] for node in current_layer]))
				self.layers.append(layer)
				self.params.extend(layer.parameters())
			
			

		

	def get_parameters(self):
		return self.params

	# This function actually defines the operation of the Neural Network
	# during feed forward.
	def forward(self, x):
		# Activation mode 0 is regular sigmoid and mode 
		# 1 is sigmoid shifted by -0.5
		if self.activation_mode == 0:
			x0 = torch.sigmoid(self.layers[0](x))
			for layer in self.layers[1:-1]:
				x0 = torch.sigmoid(layer(x0))
		else:
			x0 = torch.sigmoid(self.layers[0](x)) - 0.5
			for layer in self.layers[1:-1]:
				x0 = torch.sigmoid(layer(x0)) - 0.5


		x0 = self.reduction_matrix.mm(self.layers[-1](x0))
		return x0

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features