import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TrainingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.identity = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.identity(x)


def demonstrate_reparameterization():
    print("\n--- Section 1: Structural Reparameterization ---")

    # Initialize a training block (1 channel in/out for simplicity)
    train_block = TrainingBlock(in_channels=1, out_channels=1)

    # Get original 3x3 weights
    train_kernel = train_block.conv.weight.data.squeeze()

    # "Fuse" the identity branch into the 3x3 kernel
    # An identity branch is mathematically equivalent to a 3x3 kernel
    # where only the center pixel is 1.0 and all others are 0.0
    fused_kernel = train_kernel.clone()
    fused_kernel[1, 1] += 1.0  # Adding the 'identity' to the center of the 3x3 weight

    # Proof of Equivalence
    input_tensor = torch.randn(1, 1, 5, 5)
    output_train = train_block(input_tensor)
    output_fused = F.conv2d(input_tensor, fused_kernel.view(1, 1, 3, 3), padding=1)

    diff = torch.abs(output_train - output_fused).max().item()
    print(f"Max Difference between Train and Fused outputs: {diff:.8f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(train_kernel.numpy(), annot=True, cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Original 3x3 Training Kernel")

    sns.heatmap(fused_kernel.numpy(), annot=True, cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Fused Inference Kernel (Conv + Identity)")
    plt.tight_layout()
    plt.show()

def demonstrate_pruning():
    print("\n--- Section 2: Weight Pruning ---")

    # Create a simple Linear layer (15x15)
    layer = nn.Linear(15, 15, bias=False)

    # Fill with some random values for visualization
    with torch.no_grad():
        layer.weight.copy_(torch.randn(15, 15))

    original_weights = layer.weight.data.clone().numpy()

    # Apply L1 Unstructured Pruning (remove 50% of the smallest weights)
    prune.l1_unstructured(layer, name="weight", amount=0.5)

    # The 'weight' is now stored in 'weight_orig' and a mask is applied
    pruned_weights = layer.weight.data.numpy()

    # Calculate Sparsity
    num_zeros = np.sum(pruned_weights == 0)
    total_elements = pruned_weights.size
    sparsity = 100. * num_zeros / total_elements

    print(f"Target Sparsity: 50%")
    print(f"Actual Sparsity achieved: {sparsity:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(original_weights, cmap="RdBu", center=0, ax=axes[0])
    axes[0].set_title("Original Dense Weights")

    sns.heatmap(pruned_weights, cmap="RdBu", center=0, ax=axes[1], mask=(pruned_weights == 0))
    axes[1].set_facecolor('black')  # Black represents pruned (zeroed) connections
    axes[1].set_title("Pruned Sparse Weights (50% sparsity)")

    plt.suptitle("Weight Pruning Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demonstrate_reparameterization()

    demonstrate_pruning()