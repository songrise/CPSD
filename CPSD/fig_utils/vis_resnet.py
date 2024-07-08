import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def read_tensor(path):
    # Read the tensor from the given path
    tensor = torch.load(path)

    # Move the tensor to CPU if necessary
    if tensor.is_cuda:
        tensor = tensor.cpu()

    return tensor


def apply_pca(tensor, n_components):
    # Convert the tensor to numpy array
    tensor_np = tensor.numpy()

    # Reshape the tensor to (B, C, H*W)
    B, C, H, W = tensor_np.shape
    tensor_reshaped = tensor_np.reshape(B, C, H * W)

    # Apply PCA to each item in the batch
    pca = PCA(n_components=n_components)
    tensor_pca = np.zeros((B, n_components, H, W))
    for i in range(B):
        tensor_pca[i] = pca.fit_transform(tensor_reshaped[i].T).T.reshape(
            n_components, H, W
        )

    return tensor_pca


def process_tensor(tensor, n_components):
    # Get the first item in the batch
    tensor_item = tensor[0]

    if n_components == 3:
        # Normalize the tensor to the range [0, 1]
        tensor_normalized = (tensor_item.transpose(1, 2, 0) - tensor_item.min()) / (
            tensor_item.max() - tensor_item.min()
        )
        return tensor_normalized
    elif n_components == 1:
        # Apply jet colormap for 1 component
        tensor_colormap = plt.cm.jet(tensor_item.squeeze())
        # Normalize the colormap to the range [0, 1]
        tensor_colormap_normalized = (tensor_colormap - tensor_colormap.min()) / (
            tensor_colormap.max() - tensor_colormap.min()
        )
        return tensor_colormap_normalized
    else:
        raise ValueError(f"Unsupported number of components: {n_components}")


# Set the number of rows and columns for subplots
num_rows = 2
num_cols = 4

# Create a figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

# Iterate over the files and process the tensors
for i in range(1, 8):
    tensor_path = f"/root/autodl-tmp/CPSD/vis_out/resnet/h_shared_layer_{i}.pt"
    # tensor_path = f"/root/autodl-tmp/CPSD/vis_out/resnet/resout_layer_{i}.pt"
    tensor = read_tensor(tensor_path)
    tensor_pca = apply_pca(tensor, 3)
    tensor_processed = process_tensor(tensor_pca, 3)

    # Calculate the subplot index
    row = (i - 1) // num_cols
    col = (i - 1) % num_cols

    # Plot the processed tensor in the corresponding subplot
    axes[row, col].imshow(tensor_processed)
    axes[row, col].set_title(f"Layer {i}")
    axes[row, col].axis("off")

# Adjust the spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig(f"/root/autodl-tmp/CPSD/vis_out/resnet/processed_tensors.png")
print(f"Processed tensors saved to processed_tensors.png")
