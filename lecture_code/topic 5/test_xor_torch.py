import numpy as np
import torch
from torch.nn.parameter import Parameter

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


learning_rate = 0.1


class NeuralNet(torch.nn.Module):
	def __init__(self):
		super(NeuralNet, self).__init__()
		self.fc1 = torch.nn.Linear(2, 2, bias=True) 
		self.relu = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(2, 1, bias=False)  

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out

model = NeuralNet().to(device)

W = np.array([[1, 1], [1, 1]]).astype(np.float32) + 0.1*np.random.rand(2, 2).astype(np.float32)
c = np.array([0, -1]).astype(np.float32) + 0.1*np.random.rand(2).astype(np.float32)
u = np.array([1, -2]).astype(np.float32) + 0.1*np.random.rand(2).astype(np.float32)

model.fc1.weight = Parameter(torch.from_numpy(W))
model.fc1.bias = Parameter(torch.from_numpy(c))
model.fc2.weight = Parameter(torch.from_numpy(u))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X = torch.from_numpy(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32))
Y = torch.from_numpy(np.array([0, 1, 1, 0]).astype(np.float32))

# train the model
for j in range(10000):
	
	outputs = model(X)
	loss = criterion(outputs, Y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if (j+1) % 100 == 0: 
		print('Epoch {}, loss = {:.4f}'.format(j, loss.item()))


O = model(X)
print(O.detach().cpu().numpy())

print(model.fc1.weight.detach().numpy())
print(model.fc1.bias.detach().numpy())
print(model.fc2.weight.detach().numpy())