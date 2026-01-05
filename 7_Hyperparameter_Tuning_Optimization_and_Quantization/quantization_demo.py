import os
import time
import copy
import sys
import platform
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import QuantStub, DeQuantStub
import matplotlib.pyplot as plt
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_platform_backend():
    # Check what engines are supported by this PyTorch build
    supported_engines = torch.backends.quantized.supported_engines
    machine_arch = platform.machine().lower()

    print(f"System Architecture: {machine_arch}")
    print(f"Supported Engines: {supported_engines}")

    # Prioritize qnnpack for ARM/Apple Silicon, fbgemm for x86
    if 'qnnpack' in supported_engines and ('arm' in machine_arch or 'aarch64' in machine_arch):
        return 'qnnpack'
    elif 'fbgemm' in supported_engines:
        return 'fbgemm'
    else:
        return 'qnnpack'  # Default fallback


QUANTIZATION_BACKEND = get_platform_backend()
torch.backends.quantized.engine = QUANTIZATION_BACKEND
DEVICE = torch.device("cpu")
torch.manual_seed(42)

print(f"Running demo on: {DEVICE}")
print(f"Selected Quantization Engine: {QUANTIZATION_BACKEND.upper()}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = QuantStub()

        # A simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 14 * 14, 10)

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = DeQuantStub()

    def forward(self, x):
        # Manually quantize the input
        x = self.quant(x)

        # Process layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        # Manually de-quantize the output
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2']], inplace=True)

def get_model_size_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size_mb


def evaluate_model(model, data_loader, description):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    accuracy = 100 * correct / total
    avg_inference_time = (end_time - start_time) / len(data_loader)

    print(f"[{description}] Accuracy: {accuracy:.2f}% | Time/Batch: {avg_inference_time * 1000:.2f}ms")
    return accuracy, avg_inference_time


def plot_weights_histogram(float_model, quantized_model):
    float_weights = float_model.conv1.weight.detach().numpy().flatten()
    quant_weights = quantized_model.conv1.weight().int_repr().numpy().flatten()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(float_weights, bins=100, color='blue', alpha=0.7)
    plt.title("Float32 Weights (Original)\nContinuous Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(quant_weights, bins=100, color='orange', alpha=0.7)
    plt.title("Int8 Weights (Quantized)\nDiscrete Integers [-128, 127]")
    plt.xlabel("Quantized Integer Value")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    plt.savefig("weight_histograms.png")

if __name__ == "__main__":
    print(">>> 1. Preparing Data (MNIST)...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Check if data exists, download if not
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Using a subset for speed in this demo
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, range(1000)), batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, range(500)), batch_size=32,
                                              shuffle=False)

    print("\n>>> 2. Training Baseline Float32 Model...")
    float_model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(float_model.parameters(), lr=0.001)

    float_model.train()
    for epoch in range(1):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = float_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Training complete.")

    print("\n>>> 3. Evaluating Float32 Model...")
    float_size = get_model_size_mb(float_model)
    float_acc, float_time = evaluate_model(float_model, test_loader, "Float32 Model")
    print(f"Float32 Size: {float_size:.2f} MB")

    print(f"\n>>> 4. Performing Post-Training Quantization (Engine: {QUANTIZATION_BACKEND})...")

    quantized_model = copy.deepcopy(float_model)
    quantized_model.eval()

    print("   Step A: Fusing layers (Conv + ReLU)...")
    quantized_model.fuse_model()

    print(f"   Step B: Preparing configuration ({QUANTIZATION_BACKEND})...")
    quantized_model.qconfig = torch.ao.quantization.get_default_qconfig(QUANTIZATION_BACKEND)
    torch.ao.quantization.prepare(quantized_model, inplace=True)

    print("   Step C: Calibrating (feeding sample data)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i > 10: break
            quantized_model(images)

    print("   Step D: Converting to Int8...")
    torch.ao.quantization.convert(quantized_model, inplace=True)

    print("\n>>> 5. Evaluating Quantized Int8 Model...")
    quant_size = get_model_size_mb(quantized_model)
    quant_acc, quant_time = evaluate_model(quantized_model, test_loader, "Int8 Model")
    print(f"Int8 Size: {quant_size:.2f} MB")

    print(f"{'Metric':<20} | {'Float32':<10} | {'Int8':<10} | {'Improvement'}")
    print("-" * 65)
    print(f"{'Size (MB)':<20} | {float_size:<10.2f} | {quant_size:<10.2f} | {float_size / quant_size:.1f}x smaller")
    print(f"{'Accuracy (%)':<20} | {float_acc:<10.2f} | {quant_acc:<10.2f} | {float_acc - quant_acc:.2f}% diff")
    print("-" * 65)

    print("Displaying weight histogram...")
    plot_weights_histogram(float_model, quantized_model)

    labels = ['Float32', 'Int8']
    sizes = [float_size, quant_size]
    accuracies = [float_acc, quant_acc]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Model Size (MB)', color=color)
    bars = ax1.bar(labels, sizes, color=color, alpha=0.6, width=0.4, label='Size')
    ax1.tick_params(axis='y', labelcolor=color)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f} MB', ha='center', va='bottom')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color)
    line = ax2.plot(labels, accuracies, color=color, marker='o', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    plt.title(f"Quantization Impact ({QUANTIZATION_BACKEND})\nSize vs Accuracy")
    fig.tight_layout()
    plt.show()