import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Sample input: 3 words represented by 4-dimensional vectors
words = ["Word1", "Word2", "Word3"]
X = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 2.0, 0.0, 2.0],
                  [1.0, 1.0, 1.0, 1.0]])

# Random weight matrices for Q, K, V (4x4)
torch.manual_seed(0)  # Ensures reproducibility
W_Q = torch.rand(4, 4)
W_K = torch.rand(4, 4)
W_V = torch.rand(4, 4)

# Step 1: Calculate Q, K, V
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Step 2: Compute attention scores
d_k = Q.size(-1)  # Dimension of K
attention_scores = (Q @ K.T) / (d_k ** 0.5)

# Step 3: Apply softmax to get attention weights
attention_weights = F.softmax(attention_scores, dim=-1)

# Step 4: Multiply attention weights with V
output = attention_weights @ V

# Display results
print("Input (X):", X)
print("Attention Weights:", attention_weights)
print("Output:", output)

# Step 5: Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(attention_weights.detach().numpy(), annot=True, cmap='Blues',
            xticklabels=words, yticklabels=words)
plt.title("Attention Weights Heatmap")
plt.xlabel("Attended To")
plt.ylabel("Attending From")
plt.show()