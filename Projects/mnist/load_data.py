# Create a pytorch neural network that will classify digits
# below you can see how to load required data

import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(".", download=True, transform=transform)