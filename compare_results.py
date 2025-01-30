# compare_results.py
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load test data
df = pd.read_csv("test_data.csv")
X = df.values
# X = df.iloc[:,:].values
# X = df[["Feature_1"]].to_numpy()
# Train scikit-learn Isolation Forest
clf = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
clf.fit(X)
scores = clf.decision_function(X)

# Print scores
print("Scikit-learn anomaly scores:")
for i, score in enumerate(scores):
    print(f"Sample {i}: {score:.4f}")