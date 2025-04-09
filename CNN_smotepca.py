import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load raw data ===
X_train_raw = np.loadtxt('Train/X_train.txt')
y_train_raw = np.loadtxt('Train/y_train.txt').astype(int) - 1  # zero-indexed

X_test = np.loadtxt('Test/X_test.txt')
y_test = np.loadtxt('Test/y_test.txt').astype(int) - 1

# === Apply SMOTE to training data ===
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_raw, y_train_raw)

# === Apply PCA to reduce dimensionality (e.g. to 100 components) ===
pca = PCA(n_components=100)  # you can tune this number
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test)

# === Reshape for CNN: (samples, features, 1)
X_train = X_train_pca.reshape((X_train_pca.shape[0], X_train_pca.shape[1], 1))
X_test = X_test_pca.reshape((X_test_pca.shape[0], X_test_pca.shape[1], 1))

# === Build CNN Model ===
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(12, activation='softmax')  # 12 activity classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Train ===
model.fit(X_train, y_train_balanced, epochs=30, batch_size=64, validation_split=0.2)

# === Evaluate ===
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# === Confusion Matrix ===
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
