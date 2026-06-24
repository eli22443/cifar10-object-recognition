# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# Hyperparameters
input_size = 784
num_classes = 10

hidden_size = 500
num_epochs = 5
batch_size = 100
learning_rate = 0.001

wd = 0
p = 0.8
alpha = 0.1

use_dropout = False
use_mixup = True

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
										   train=True,
										   transform=transforms.ToTensor(),
										   download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
										  train=False,
										  transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False)

images, labels = next(iter(train_loader))


fig, axs = plt.subplots(2, 5)
for ii in range(2):
	for jj in range(5):
		idx = 5 * ii + jj
		axs[ii, jj].imshow(images[idx].squeeze())
		axs[ii, jj].set_title(labels[idx].item())
		axs[ii, jj].axis('off')
plt.show()


# Fully connected neural network
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) 
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)  
		# dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, x):
		out = self.fc1(x)
		if use_dropout:
			out = self.dropout(out)
		out = self.relu(out)
		out = self.fc2(out)
		return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
if use_mixup:
	def criterion(input, target):
		return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()

def mixup(input, target, gamma):
	indices = torch.randperm(input.size(0))
	return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)

def partial_mixup(input, gamma, indices):
	if input.size(0) != indices.size(0):
		raise RuntimeError("mismatch")

	perm_input = input[indices]
	return input.mul(gamma).add(perm_input, alpha=1-gamma)


# weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
	weight_decay=wd)

# train the model
model.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, input_size).to(device)
		labels = nn.functional.one_hot(labels.to(device), num_classes)

		if use_mixup:
			images, labels = mixup(images, labels, 
				np.random.beta(alpha, alpha))

		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
