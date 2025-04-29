import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv('mushroom_data.csv')

# Pisahkan fitur dan label
X = df.drop(columns=['class'])  # Semua kolom kecuali class
y = df['class']  # Kolom target edible/poisonous

# Encode semua fitur dan label karena datanya berbentuk kategori
le_X = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    le_X[column] = le  # Simpan encoder fitur (opsional, kalau mau inverse transform)

# Encode label (class)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Buat model Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Simpan model
joblib.dump(model, 'decision_tree_model.pkl')

# Simpan LabelEncoders jika nanti mau digunakan untuk decoding hasil prediksi
joblib.dump(le_X, 'feature_encoders.pkl')
joblib.dump(le_y, 'label_encoder.pkl')

print("Model dan encoder berhasil disimpan!")
