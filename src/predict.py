import cv2
import numpy as np
import pickle

# Load model
with open("models/model.pkl", "rb") as f:
    model, pca = pickle.load(f)

# Test image path
image_path = "test.jpg"

img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

# SAME preprocessing as training
img = cv2.resize(img, (64, 64))
img = img.flatten().reshape(1, -1)

# Apply PCA
img = pca.transform(img)

# Predict
prediction = model.predict(img)

print("🎯 Prediction:", prediction[0])