import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
import pickle
from PIL import Image

# Load model MobileNetV2 (fitur extractor)
feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128,128,3))

def load_images_from_folder(folder):
    X, y = [], []
    for label in ['edible', 'poisonous']:
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize((128,128))
                img_array = image.img_to_array(img)
                img_array = preprocess_input(img_array)
                features = feature_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
                X.append(features[0])
                y.append(0 if label == 'edible' else 1)
            except Exception as e:
                print(f"Gagal memproses gambar {filename}: {e}")
    return np.array(X), np.array(y)

# Load data & ekstrak fitur
X, y = load_images_from_folder('dataset')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih Decision Tree
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)
print("Decision Tree Accuracy:", clf.score(X_test, y_test))

# Simpan model ke file .pkl
with open('mushroom_dtc_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Model tersimpan ke 'mushroom_dtc_model.pkl'")
