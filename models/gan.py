import torch
import torch.nn as nn


class Generator(nn.Module):

	def __init__(self):

		super(Generator,self).__init__()

		self.generator = nn.Sequential(
			nn.Linear(100,128),
			nn.ReLU(True),
			nn.Linear(128,784),
			nn.Sigmoid()
		)

	def forward(self,z):

		out = self.generator(z)

		return out

class Discriminator(nn.Module):

	def __init__(self):

		super(Discriminator,self).__init__()

		self.discriminator = nn.Sequential(
			nn.Linear(784,128),
			nn.ReLU(True),
			nn.Linear(128,1),
			nn.Sigmoid()
		)

	def forward(self, x):

		x = self.discriminator(x)

		return x

