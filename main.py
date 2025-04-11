import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load scaled features
scaled_df = pd.read_csv("tsla_scaled_features.csv", index_col="Date")

# Apply PCA
pca = PCA(n_components=0.95)  # Keep components explaining 95% of variance
pca_result = pca.fit_transform(scaled_df)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(pca_result, index=scaled_df.index, 
                      columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Component loadings (how features contribute to PCs)
loadings = pd.DataFrame(pca.components_.T, columns=pca_df.columns, 
                        index=scaled_df.columns)

# Save PCA results
pca_df.to_csv("tsla_pca_results.csv")
loadings.to_csv("tsla_pca_loadings.csv")

# Visualizations
# 1. Scree plot (explained variance)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, "bo-", label="Explained Variance")
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "ro-", label="Cumulative Variance")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot for Tesla Stock and Sentiment PCA")
plt.legend()
plt.grid()
plt.savefig("scree_plot.png")
plt.close()

# 2. Loadings heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
plt.title("PCA Loadings: Feature Contributions to Principal Components")
plt.tight_layout()
plt.savefig("loadings_heatmap.png")
plt.close()

# Display results
print("PCA Explained Variance Ratio:")
print(pd.Series(explained_variance, index=[f"PC{i+1}" for i in range(len(explained_variance))]))
print("\nCumulative Variance Explained:")
print(pd.Series(cumulative_variance, index=[f"PC{i+1}" for i in range(len(cumulative_variance))]))
print("\nPCA Loadings (Feature Contributions):")
print(loadings)
print("\nPCA Results Preview:")
print(pca_df.head())
