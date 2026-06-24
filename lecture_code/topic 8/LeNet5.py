import torch
import torch.nn as nn

class LeNet5(nn.Module):
	def __init__(self, output_dim):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)

		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, output_dim)

		self.pool = nn.AvgPool2d(2, 2)
		self.act = nn.Sigmoid()

	def forward(self, x):
		x = self.act(self.pool(self.conv1(x)))
		x = self.act(self.pool(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = self.act(self.fc1(x))
		x = self.act(self.fc2(x))
		x = self.fc3(x)
		return x


