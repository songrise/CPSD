# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the tensor from the specified file path
tensor_path = "/root/autodl-tmp/CPSD/vis_out/traj/alpha_cum.pt"
alphas = torch.load(tensor_path)

# Extract samples every 20 steps, assuming there are sufficient elements.
stride = 20
num_samples = 50
indices = [i * stride for i in range(num_samples)]

# Ensure indices are within the bounds of the tensor size
if indices[-1] >= len(alphas):
    raise ValueError(
        "The tensor does not have enough elements to sample 50 values with the given stride."
    )

# Sample the tensor using the calculated indices
sampled_alphas = alphas[indices]

# Convert tensor to numpy array for plotting
sampled_alphas_np = sampled_alphas.numpy()

# # Plot the trend
# plt.figure(figsize=(10, 6))
# plt.plot(sampled_alphas_np, marker="o", linestyle="-", label="Sampled Alpha Trend")
# plt.xlabel("Sample Index")
# plt.ylabel("Alpha Value")
# plt.title("Trend of Sampled Alpha Values (every 20 steps)")
# plt.legend()
# plt.grid(True)

# Show the plot
# plt.show()


def A_t(t, alphas):

    return np.sqrt(alphas[t - 1]) / np.sqrt(alphas[t])


def B_t(t, alphas):

    return np.sqrt(1 - alphas[t - 1]) - np.sqrt(alphas[t] * (1 - alphas[(t - 1)]))


def cum_score(low_t, hi_t, alphas):
    res = 0
    for k in range(low_t + 1, hi_t):
        for j in range(low_t + 1, k - 1):
            res += A_t(j, alphas) * B_t(k, alphas)
    return res


content_scores = []
style_scores = []


for tau in range(0, 50):
    crt_content = cum_score(50 - tau, 50, sampled_alphas)
    crt_style = cum_score(0, tau, sampled_alphas)
    content_scores.append(crt_content)
    style_scores.append(crt_style)

plt.figure(figsize=(10, 6))
plt.plot(content_scores, marker="o", linestyle="-", label="Content Score")
plt.plot(style_scores, marker="o", linestyle="-", label="Style Score")
plt.xlabel("Tau Value")
plt.ylabel("Cumulative Score")
plt.title("Cumulative Content and Style Scores")
plt.legend()
plt.grid(True)
plt.show()
