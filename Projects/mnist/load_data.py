# Create a pytorch neural network that will classify digits
# below you can see how to load required data
# you can iterate over the dataset to get data-label pairs

import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(".", download=True, transform=transform)