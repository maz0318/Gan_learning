import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def set_seed(seed):
	'''
	Fix immediate seed to repeat experiment
	:param seed: An integer
	:return:
	'''
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


def loadMNIST(batch_size):

	transforms_op = transforms.Compose([transforms.ToTensor()])

	train_set = MNIST('./data', train=True, transform=transforms_op , download=True)
	test_set = MNIST('./data', train=False, transform=transforms_op)

	train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
	test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
	return train_data, test_data
