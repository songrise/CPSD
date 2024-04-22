#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the parameters for the two Gaussian distributions
mu1 = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])

mu2 = np.array([3, 3])
cov2 = np.array([[1, 0.5], [0.5, 1]])

# Define the mixture weights
w1, w2 = 0.4, 0.6

# Generate a grid of points
x, y = np.meshgrid(np.linspace(-5, 8, 20), np.linspace(-5, 8, 20))
pos = np.dstack((x, y))

# Evaluate the mixture of Gaussian distributions
pdf1 = multivariate_normal.pdf(pos, mean=mu1, cov=cov1)
pdf2 = multivariate_normal.pdf(pos, mean=mu2, cov=cov2)
pdf = w1 * pdf1 + w2 * pdf2

# Plot the PDF with shading
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(x, y, pdf, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Mixture of Two 2D Gaussian Distributions')
ax.grid(True)

# Calculate the score function with respect to the parameters
def score_function(pos, mu1, mu2, cov1, cov2, w1, w2):
    pdf1 = multivariate_normal.pdf(pos, mean=mu1, cov=cov1)
    pdf2 = multivariate_normal.pdf(pos, mean=mu2, cov=cov2)
    pdf = w1 * pdf1 + w2 * pdf2

    score_mu1 = w1 * np.sum(np.linalg.inv(cov1) @ (pos - mu1)[:, :, np.newaxis] * (pdf1 / pdf)[:, :, np.newaxis], axis=(0, 1))
    score_mu2 = w2 * np.sum(np.linalg.inv(cov2) @ (pos - mu2)[:, :, np.newaxis] * (pdf2 / pdf)[:, :, np.newaxis], axis=(0, 1))
    
    score_cov1 = 0.5 * w1 * np.sum((np.linalg.inv(cov1) - np.linalg.inv(cov1) @ (pos - mu1)[:, :, np.newaxis] @ (pos - mu1)[:, np.newaxis, :] @ np.linalg.inv(cov1)) * (pdf1 / pdf)[:, :, np.newaxis, np.newaxis], axis=(0, 1))
    score_cov2 = 0.5 * w2 * np.sum((np.linalg.inv(cov2) - np.linalg.inv(cov2) @ (pos - mu2)[:, :, np.newaxis] @ (pos - mu2)[:, np.newaxis, :] @ np.linalg.inv(cov2)) * (pdf2 / pdf)[:, :, np.newaxis, np.newaxis], axis=(0, 1))
    
    score_w1 = np.sum(pdf1 / pdf)
    score_w2 = np.sum(pdf2 / pdf)
    
    return score_mu1, score_mu2, score_cov1, score_cov2, score_w1, score_w2

# Calculate the score function for the given parameters
score_mu1, score_mu2, score_cov1, score_cov2, score_w1, score_w2 = score_function(pos, mu1, mu2, cov1, cov2, w1, w2)

print("Score function with respect to the parameters:")
print("mu1:", score_mu1)
print("mu2:", score_mu2)
print("cov1:", score_cov1)
print("cov2:", score_cov2)
print("w1:", score_w1)
print("w2:", score_w2)

plt.tight_layout()
plt.show()