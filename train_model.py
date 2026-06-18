import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Input, Dropout, Flatten, Attention
from tensorflow.keras.utils import to_categorical

# =========================
# LOAD DATA
# =========================
train = pd.read_csv("C:/Users/Asus/OneDrive/Desktop/wearable-health-ai/dataset/train.csv", header=None)
test = pd.read_csv("C:/Users/Asus/OneDrive/Desktop/wearable-health-ai/dataset/test.csv", header=None)

data = pd.concat([train, test])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# normalize
X = X / np.max(X)

# reshape
X = X.reshape(X.shape[0], X.shape[1], 1)

# one-hot encode
y = to_categorical(y)

# =========================
# CLASS DISTRIBUTION
# =========================
labels = np.argmax(y, axis=1)

plt.figure()
plt.hist(labels, bins=5)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
inputs = Input(shape=(X.shape[1], 1))

x = Conv1D(64, 5, activation='relu')(inputs)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

lstm_out = LSTM(128, return_sequences=True)(x)

attention = Attention()([lstm_out, lstm_out])

x = Flatten()(attention)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(5, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# =========================
# ACCURACY PRINT
# =========================
print("\n🔥 FINAL RESULTS")
print(f"Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

loss, test_acc = model.evaluate(X_test, y_test)
print(f"🚀 Test Accuracy: {test_acc*100:.2f}%")

# =========================
# 📈 ACCURACY GRAPH
# =========================
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

# =========================
# 📉 LOSS GRAPH
# =========================
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()

# =========================
# 🤯 CONFUSION MATRIX
# =========================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 📊 PRECISION / RECALL
# =========================
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred_classes))

# =========================
# 🔥 ROC CURVE
# =========================
y_test_bin = label_binarize(y_true, classes=[0,1,2,3,4])

plt.figure()

for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    plt.plot(fpr, tpr, label=f"Class {i}")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =========================
# 📊 PROBABILITY GRAPH
# =========================
sample = X_test[0].reshape(1, X_test.shape[1], 1)
probs = model.predict(sample)[0]

plt.figure()
plt.bar(range(5), probs)
plt.title("Prediction Probability")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.show()

# =========================
# 📉 OVERFITTING GRAPH
# =========================
plt.figure()
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Validation")

plt.fill_between(
    range(len(history.history['accuracy'])),
    history.history['accuracy'],
    history.history['val_accuracy'],
    alpha=0.2
)

plt.title("Overfitting Check")
plt.legend()
plt.show()

# =========================
# SAVE MODEL
# =========================
model.save("ecg_lstm_model.h5")

print("✅ EVERYTHING DONE (MODEL + ALL GRAPHS)")