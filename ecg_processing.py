import neurokit2 as nk
import numpy as np

def extract_features(ecg):

    # Process ECG
    signals, info = nk.ecg_process(ecg, sampling_rate=100)

    # HR
    heart_rate = int(np.mean(signals["ECG_Rate"]))

    # HRV Features
    hrv = nk.hrv(signals, sampling_rate=100)

    rmssd = float(hrv["HRV_RMSSD"].values[0])
    sdnn = float(hrv["HRV_SDNN"].values[0])

    return {
        "heart_rate": heart_rate,
        "rmssd": rmssd,
        "sdnn": sdnn
    }


def detect_stress(features):

    # Rule-based + ML logic hybrid
    if features["rmssd"] < 20 or features["sdnn"] < 30:
        return "HIGH"
    elif features["rmssd"] < 40:
        return "MEDIUM"
    else:
        return "LOW"


def detect_emotion(features):

    hr = features["heart_rate"]

    if hr > 100:
        return "ANXIOUS 😰"
    elif hr < 60:
        return "CALM 😊"
    elif 60 <= hr <= 90:
        return "NORMAL 🙂"
    else:
        return "STRESSED 😡"