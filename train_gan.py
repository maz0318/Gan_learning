import torch
import os
import argparse
from utils.general_tools import *
from models.gan import *
from torchvision.utils import save_image

# 设置随机种子
set_seed(21)

# 选择cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

	if not os.path.exists(args.result_path):
		os.mkdir(args.result_path)
	# 载入数据
	train_data, test_data = loadMNIST(args.batch_size)

	# 创建模型
	G = Generator().to(device)
	D = Discriminator().to(device)


	# 优化器
	G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)
	D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)


	# 训练
	total_step = len(train_data)
	for epoch in range(1,100000):
		epoch_G_loss = 0
		epoch_D_loss = 0
		for images, _ in train_data:
			batch_size = images.size(0)
			images = images.reshape(batch_size,-1).to(device)

			# ================================================================== #
			#                      Train the discriminator                       #
			# ================================================================== #

			# Using real data
			outputs_real = D(images)

			# Using fake data
			z = torch.randn(batch_size, args.latent_dim).to(device)
			fake_images = G(z)
			outputs_fake = D(fake_images)

			D_loss = - torch.mean(torch.log(outputs_real+1e-10) + torch.log(1. - outputs_fake+1e-10))
			epoch_D_loss += D_loss.item()
			D_optimizer.zero_grad()
			D_loss.backward()
			D_optimizer.step()

			# ==============================================total_step = len(train_data)==================== #
			#                        Train the generator                         #
			# ================================================================== #


			z = torch.randn(batch_size, args.latent_dim).to(device)
			fake_images = G(z)
			outputs = D(fake_images)

			G_loss = -torch.mean(torch.log(outputs+1e-10))
			epoch_G_loss += G_loss.item()
			G_optimizer.zero_grad()
			G_loss.backward()
			G_optimizer.step()
		print("*** epoch{}, G_loss:{:.4f}, D_loss:{:.4f} ***".format(epoch,epoch_G_loss/total_step,epoch_D_loss/total_step))
		if epoch == 1:
			images = images.reshape(batch_size,1,28,28)
			save_image(images,args.result_path+'real_images.png')
		if epoch %100 == 0:
			fake_images = fake_images.reshape(batch_size,1,28,28)
			save_image(fake_images,args.result_path+'fake_images_{}.png'.format(epoch))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size',type=int,
						default=64,
						help="一次训练的样本个数")

	parser.add_argument('--lr',type=float,
						default=1e-3,
						help="学习率")
	parser.add_argument('--latent_dim',type=int,
						default=100,
						help="噪声的维度")
	parser.add_argument('--result_path',type=str,
						default='./result/gan/')

	args = parser.parse_args()
	main(args)
