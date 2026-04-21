import os
import cv2
import numpy as np

def load_images(data_path):
    X = []
    y = []

    max_images = 5000   # 🔥 limit
    count = 0

    for label in os.listdir(data_path):
        folder_path = os.path.join(data_path, label)

        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):

            if count >= max_images:
                break   # 🔥 stop loading more images

            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            # 🔥 reduce size (VERY IMPORTANT)
            img = cv2.resize(img, (64, 64))

            img = img.flatten()

            X.append(img)
            y.append(label)

            count += 1   # 🔥 increase count

    return np.array(X), np.array(y)