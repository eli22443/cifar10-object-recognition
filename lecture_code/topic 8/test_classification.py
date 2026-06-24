import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from tqdm import tqdm

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True



# data handling
root = './data'
num_classes = 10
train_data = datasets.CIFAR10(root=root, train=True, download=True)

means = train_data.data.mean(axis=(0, 1, 2)) / 255.0
stds = train_data.data.std(axis=(0, 1, 2)) / 255.0
print(f'means = {means}, stds = {stds}')

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=means, std=stds)
	])


train_data = datasets.CIFAR10(root=root, train=True, 
	download=True, transform=transform)

test_data = datasets.CIFAR10(root=root, train=False, 
	download=True, transform=transform)


valid_ratio = 0.9
n_train_examples = int(len(train_data) * valid_ratio)
n_valid_examples = len(train_data) - n_train_examples


train_data, valid_data = data.random_split(train_data,
	[n_train_examples, n_valid_examples])
print(f'#train = {len(train_data)}, #valid = {len(valid_data)}')

batch_size = 256
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = data.DataLoader(valid_data, batch_size=batch_size)
test_loader = data.DataLoader(test_data, batch_size=batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model_name = 'ResNet18'
if model_name == 'LeNet5':
	from LeNet5 import LeNet5
	model = LeNet5(output_dim = num_classes)
elif model_name == 'VGG11':
	from VGG import VGG
	model = VGG(output_dim=num_classes, vgg_cfg=model_name)
elif model_name == 'ResNet18':
	from ResNet import BasicBlock, ResNet

	cfg = (BasicBlock,
		[2, 2, 2, 2],
		[64, 128, 256, 512]
		)

	# cfg = (BasicBlock,
	# 	[3, 4, 6, 3],
	# 	[64, 128, 256, 512]
	# 	)

	model = ResNet(config=cfg, output_dim=num_classes)
model = model.to(device)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has #parameters = {count_parameters(model)}')

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# train + eval
def calc_acc(y_pred, y):
	top_pred = y_pred.argmax(1, keepdim=True)
	correct = top_pred.eq(y.view_as(top_pred)).sum()
	acc = correct.float() / y.shape[0]

	return acc


def train(model, loader, optimizer, criterion):
	epoch_loss, epoch_acc = 0, 0

	model.train()

	for (x, y) in tqdm(loader):
		optimizer.zero_grad()

		y_pred = model(x.to(device))
		loss = criterion(y_pred, y.to(device))
		loss.backward()
		optimizer.step()

		acc = calc_acc(y_pred, y.to(device))

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion):
	epoch_loss, epoch_acc = 0, 0

	model.eval()

	with torch.no_grad():
		for (x, y) in tqdm(loader):
			y_pred = model(x.to(device))
			loss = criterion(y_pred, y.to(device))
			acc = calc_acc(y_pred, y.to(device))

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(loader), epoch_acc / len(loader)


epochs = 25
best_valid_loss = float('inf')
for epoch in range(epochs):
	train_loss, train_acc = train(model, train_loader, optimizer, criterion)
	valid_loss, valid_acc = evaluate(model, valid_loader, criterion)

	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), model_name + '.pt')

	print(f'Train loss {train_loss:.3f} | Train acc {train_acc*100:.2f}%')
	print(f'Valid loss {valid_loss:.3f} | Valid acc {valid_acc*100:.2f}%')


