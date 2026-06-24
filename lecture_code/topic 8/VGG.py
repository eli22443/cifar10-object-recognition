import torch
import torch.nn as nn

vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG(nn.Module):

	def __init__(self, output_dim, vgg_cfg):
		super().__init__()

		self.features = self.get_layers(vgg_cfg)

		self.avgpool = nn.AdaptiveAvgPool2d(7)
		self.dropout = nn.Dropout(0.5)
		self.relu = nn.ReLU(inplace=True)
		self.classifier = nn.Sequential(
			self.dropout,
			nn.Linear(512 * 7 * 7, 4096),
			self.relu,
			self.dropout,
			nn.Linear(4096, 4096),
			self.relu,
			nn.Linear(4096, output_dim)
			)

	def get_layers(self, vgg_cfg):
		if vgg_cfg == 'VGG11':
			cfg = vgg11_config

		layers, in_channels = [], 3

		for c in cfg:
			if c == 'M':				# MaxPool
				layers += [nn.MaxPool2d(kernel_size=2)]

			else:

				conv2d = nn.Conv2d(in_channels, c, 
					kernel_size=3, padding=1)
				layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = c
		return nn.Sequential(*layers) 


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x