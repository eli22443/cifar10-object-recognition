# dataset of family names in different countries; train an RNN classifier
import torch
import torch.nn as nn
from io import open
import glob
import os
import unicodedata
import string
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def findFiles(path): return glob.glob(path)


def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters)


def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]


# convert letter to index (number)
def letterToIndex(letter):
	return all_letters.find(letter)


# one-hot encoding of letters
def letterToTensor(letter):
	tensor = torch.zeros(1, n_letters)
	tensor[0][letterToIndex(letter)] = 1
	return tensor


def lineToTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		# self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2h = nn.Linear(input_size, hidden_size)
		self.h2h = nn.Linear(hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)
		
		self.tanh = nn.Tanh()
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		# combined = torch.cat((input, hidden), 1)
		# hidden = self.i2h(combined)
		hidden = self.tanh(self.i2h(input) + self.h2h(hidden))
		output = self.softmax(self.h2o(hidden))

		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


def categoryFromOutput(output):
	top_n, top_i = output.topk(1)
	category_i = top_i[0].item()
	return all_categories[category_i], category_i


def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = torch.tensor([all_categories.index(category)], 
		dtype=torch.long)
	line_tensor = lineToTensor(line)
	return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
	rnn.train()

	hidden = rnn.initHidden()

	optimizer.zero_grad()

	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	loss = criterion(output, category_tensor)
	loss.backward()

	optimizer.step()

	return output, loss.item()


def evaluate(line_tensor):

	rnn.eval()
	with torch.no_grad():
		hidden = rnn.initHidden()

		for i in range(line_tensor.size()[0]):
			output, hidden = rnn(line_tensor[i], hidden)

		return output

###############
# data handling
###############
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print(findFiles('data/names/*.txt'))
print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

for filename in findFiles('data/names/*.txt'):
	category = os.path.splitext(os.path.basename(filename))[0]
	all_categories.append(category)
	lines = readLines(filename)
	category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines['Italian'][:5])

####################
# names into tensors
####################

print(letterToTensor('J'))
print(lineToTensor('Jones').size())


################
# neural network
################
n_hidden = 128
learning_rate = 1e-4
n_iters = 100000
# n_iters = 1000
print_every = 5000
plot_every = 1000

rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output.shape)
print(output)
print(categoryFromOutput(output))


for i  in range(10):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	print(f'category = {category}, line = {line}')


current_loss = 0
all_losses = []

# train the model
for iter in range(1, n_iters + 1):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	output, loss = train(category_tensor, line_tensor)
	current_loss += loss

	if iter % print_every == 0:
		guess, guess_i = categoryFromOutput(output)
		correct = 'v' if guess == category else f'x {category}'

		print('%d %d%% %.4f %s / %s %s' % (iter, iter / n_iters * 100, 
			loss, line, guess, correct))

	if iter % plot_every == 0:
		all_losses.append(current_loss / plot_every)
		current_loss = 0


plt.figure()
plt.plot(all_losses)


# evaluate the results
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

for i in range(n_confusion):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	output = evaluate(line_tensor)
	guess, guess_i = categoryFromOutput(output)
	category_i = all_categories.index(category)
	confusion[category_i][guess_i] += 1

for i in range(n_categories):
	confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
