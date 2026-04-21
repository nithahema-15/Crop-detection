from preprocess import load_images
from pca import reduce_features
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import joblib

print("🔥 Training started...")

# Load data
X, y = load_images("data/raw")
print("Data loaded")
print("Shape:", X.shape)

# Apply PCA
X, pca = reduce_features(X)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# 🔥 SAVE MODEL (THIS IS YOUR MISSING PART)
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump((model, pca), f)

print("✅ Model saved successfully!")

# Save trained model
joblib.dump(model, "model.pkl")

# Save PCA transformer
joblib.dump(pca, "pca.pkl")

print("✅ Model and PCA saved successfully!")