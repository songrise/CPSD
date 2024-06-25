# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F


def visualize_cross_attention_maps(file_path, N, b, tokens):
    # Load the attention probabilities tensor
    attention_probs = torch.load(file_path)
    attention_probs = attention_probs.detach().cpu()

    # Assuming the tensor shape is [B, H*W, L]
    B, HW, L = attention_probs.shape
    H = W = int(np.sqrt(HW))  # Assuming H*W is a perfect square

    # Ensure batch index b is within the valid range
    if b >= B:
        raise ValueError(f"Batch index b={b} is out of range. Tensor has {B} batches.")

    # Reshape attention maps to [B, H, W, L]
    attention_probs = attention_probs.reshape(B, H, W, L)
    n_sample = 6
    n_head = B // n_sample
    b = n_head * 4 - 1 + b  # condtional style is the 4th
    # Visualize attention maps for the selected batch and first N text tokens
    fig, axs = plt.subplots(1, N, figsize=(N * 4, 4))

    for i in range(N):
        attn_map = (
            attention_probs[b, :, :, i].detach().numpy()
        )  # Get attention map for the i-th text token in batch b
        attn_map_tensor = (
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dimensions
        upsampled_attn_map = (
            F.interpolate(attn_map_tensor, size=(256, 256), mode="nearest")
            .squeeze()
            .numpy()
        )
        im = axs[i].imshow(upsampled_attn_map, cmap="coolwarm", interpolation="nearest")
        axs[i].set_title(f"{tokens[i]}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_self_attention_maps(file_path, N, b, tokens):
    # Load the attention probabilities tensor
    attention_probs = torch.load(file_path)
    attention_probs = attention_probs.detach().cpu()

    # Assuming the tensor shape is [B, H*W, L]
    B, HW, L = attention_probs.shape
    H = W = int(np.sqrt(HW))  # Assuming H*W is a perfect square

    # Ensure batch index b is within the valid range
    if b >= B:
        raise ValueError(f"Batch index b={b} is out of range. Tensor has {B} batches.")

    # Reshape attention maps to [B, H, W, L]
    attention_probs = attention_probs.reshape(B, H, W, L)
    n_sample = 6
    n_head = B // n_sample
    b = n_head * 5 - 1 + b  # condtional style is the 4th
    # Visualize attention maps for the selected batch and first N text tokens
    fig, axs = plt.subplots(1, N, figsize=(N * 4, 4))

    for i in range(N):
        attn_map = (
            attention_probs[b, :, :, i].detach().numpy()
        )  # Get attention map for the i-th text token in batch b
        attn_map_tensor = (
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dimensions
        upsampled_attn_map = (
            F.interpolate(attn_map_tensor, size=(256, 256), mode="nearest")
            .squeeze()
            .numpy()
        )
        im = axs[i].imshow(upsampled_attn_map, cmap="coolwarm", interpolation="nearest")
        axs[i].set_title(f"{tokens[i]}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


# Path to the attention probabilities tensor file
file_path = "/root/autodl-tmp/CPSD/vis_out/attn/attention_probs_layer_5_cross_32.pt"
# file_path = "/root/autodl-tmp/CPSD/vis_out/attn/attention_probs_layer_7_cross_64.pt"

# Number of text tokens to visualize
N = 12  # Change this value as needed
b = 2  # Change this value as needed
# Check if the file exists
prompt = "[CLS] A impressionist painting of a boat by Claude Monet".split(" ")
prompt += ["pad", "pad", "pad"]
visualize_cross_attention_maps(file_path, N, b, prompt)

# %%
