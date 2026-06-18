import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "ecg_lstm_model.h5")

model = load_model(model_path)

def explain_ecg(signal):

    signal = np.array(signal)
    X = signal[:200].reshape(1,200,1)

    background = np.random.normal(0,1,(50,200,1))

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X)

    importance = shap_values[0][0].reshape(200)

    plt.figure(figsize=(10,4))
    plt.plot(signal)
    plt.scatter(range(200), signal, c=importance, cmap='jet')

    plt.colorbar()
    plt.title("Explainable AI ECG")

    file = "xai_output.png"
    plt.savefig(file)
    plt.close()

    return file