from src.train import *
from src.predict import predict

# test prediction
result, conf = predict("test.jpg")

print("Prediction:", result)
print("Confidence:", conf)

