import os
from PIL import Image
import numpy as np
import pickle as pkl
# Data Preprocessing
dir_ = "DATASET/train/FAKE/"
features_train = []
labels_train = []

# Process FAKE images
for filename in os.listdir(dir_):
    try:
        Img = Image.open(os.path.join(dir_, filename)).convert('RGB')
        Img = Img.resize((32, 32))  # Resize to 32x32
        img_array = np.array(Img)  # Convert to NumPy array
        features_train.append(img_array)
        labels_train.append([1, 0])  # Fake
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Process REAL images
dir__ = "DATASET/train/REAL/"
for filename in os.listdir(dir__):
    try:
        Img_ = Image.open(os.path.join(dir__, filename)).convert('RGB')
        Img_ = Img_.resize((32, 32))  # Resize to 32x32
        img_array = np.array(Img_)  # Convert to NumPy array
        features_train.append(img_array)
        labels_train.append([0, 1])  # Real
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Process test FAKE images
dir_test = "DATASET/test/FAKE/"
features_test = []
labels_test = []
for filename in os.listdir(dir_test):
    try:
        Img = Image.open(os.path.join(dir_test, filename)).convert('RGB')
        Img = Img.resize((32, 32))  # Resize to 32x32
        img_array = np.array(Img)  # Convert to NumPy array
        features_test.append(img_array)
        labels_test.append([1, 0])  # Fake
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Process test REAL images
dir__test = "DATASET/test/REAL/"
for filename in os.listdir(dir__test):
    try:
        Img_ = Image.open(os.path.join(dir__test, filename)).convert('RGB')
        Img_ = Img_.resize((32, 32))  # Resize to 32x32
        img_array = np.array(Img_)  # Convert to NumPy array
        features_test.append(img_array)
        labels_test.append([0, 1])  # Real
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Normalize and reshape data
features_train = np.array(features_train) / 255.0
features_train = np.reshape(features_train, (features_train.shape[0], 32, 32, 3))
features_train = features_train.astype('float32')

features_test = np.array(features_test) / 255.0
features_test = np.reshape(features_test, (features_test.shape[0], 32, 32, 3))
features_test = features_test.astype('float32')

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

# Print shapes
print("Train Features Shape:", features_train.shape)
print("Test Features Shape:", features_test.shape)
print("Train Labels Shape:", labels_train.shape)
print("Test Labels Shape:", labels_test.shape)

with open("train_x.pkl", "wb") as f1:
    pkl.dump(features_train, f1)
with open("test_x.pkl", "wb") as f2:
    pkl.dump(features_test, f2)
with open("train_y.pkl", "wb") as f3:
    pkl.dump(labels_train, f3)
with open("test_y.pkl", "wb") as f4:
    pkl.dump(labels_test, f4)