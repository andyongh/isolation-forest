# # generate_data.py
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_blobs

# # Generate synthetic data
# n_samples = 1000
# n_features = 5
# X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=1, cluster_std=1.0, random_state=42)

# # Add some outliers
# outliers = np.random.uniform(low=-10, high=10, size=(20, n_features))
# X = np.vstack([X, outliers])

# # Save to CSV
# df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
# df.to_csv("test_data.csv", index=False)
# print("Test data saved to 'test_data.csv'")

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate normal data (non-anomalous)
n_samples = 1000  # Number of normal data points
n_features = 1    # Number of features

# Generate data from a normal distribution
mean = 1
std_dev = 1
data = np.random.normal(loc=mean, scale=std_dev, size=(n_samples, n_features))

n_outliers = int(0.01 * n_samples)
outliers = np.random.uniform(low=20, high=30, size=(n_outliers, n_features))

# # Generate outlier data (1% of total samples)
# n_outliers = int(0.01 * n_samples)
# outlier_mean = 5  # Shifted mean for outliers
# outlier_std_dev = 1.5
# outliers = np.random.normal(loc=outlier_mean, scale=outlier_std_dev, size=(n_outliers, n_features))

# Combine normal data with outliers
data_with_outliers = np.vstack((data, outliers))
np.random.shuffle(data_with_outliers)
data_with_outliers = data_with_outliers.reshape(-1, 1)

# Create DataFrame
columns = [f'Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(data_with_outliers, columns=columns)

# Save to CSV
df.to_csv("test_data.csv", index=False)

print("Normal data with 1% outliers saved to test_data.csv")
