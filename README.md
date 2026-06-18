# AI-Based-Emotion-Detection-using-HRV
An intelligent health monitoring system that predicts human emotional states using Heart Rate Variability (HRV) signals and Machine Learning.


##  Project Overview

This project focuses on detecting human emotions such as:

* 😊 Calm
* 🙂 Normal
* 😡 Stressed
* 😰 Anxious

using physiological features like:

* Heart Rate (HR)
* RMSSD
* SDNN
* LF/HF Ratio
* pNN50

The system uses Machine Learning models with hyperparameter tuning and is deployed using a FastAPI backend for real-time predictions.

---

##  Features

*  End-to-End Machine Learning Pipeline
*  Feature Engineering on HRV Data
*  High Accuracy using Ensemble Models
*  Hyperparameter Tuning (GridSearchCV)
*  Cross-Validation for Model Reliability
*  Real-Time Prediction using FastAPI
*  Research-level Graphs & Reports

---

##  Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Backend API:** FastAPI
* **Model Saving:** Joblib

---

##  Project Structure

```
wearable-health-ai/
│
├── dataset/
│   ├── your_data.csv
│   └── train_model.py
│
├── backend/
│   └── main.py
│
├── outputs/
│   ├── graphs/
│   ├── reports/
│   └── model/
│
└── requirements.txt
```

---

##  Installation

```bash
git clone https://github.com/your-username/emotion-detection-hrv-ai.git
cd emotion-detection-hrv-ai
pip install -r requirements.txt
```

---

##  Run Training

```bash
cd dataset
python train_model.py
```

---

##  Run Backend API

```bash
cd backend
uvicorn main:app --reload
```

Open browser:

```
http://127.0.0.1:8000/docs
```

---

##  Outputs Generated

###  Graphs

* Distribution Plot
* Pair Plot
* Confusion Matrix
* ROC Curve
* Feature Importance

###  Reports

* Accuracy Report
* Classification Report

###  Model

* Trained ML Model (`emotion_model.pkl`)
* Scaler (`scaler.pkl`)

---

##  Example API Request

```json
{
  "heart_rate": 100,
  "rmssd": 20,
  "sdnn": 25,
  "lf_hf": 2.0,
  "pnn50": 15
}
```

### Response:

```json
{
  "emotion": "STRESSED 😡"
}
```

---

##  Use Cases

* Wearable Health Devices
* Stress Monitoring Systems
* Mental Health Analysis
* Smart Healthcare Applications

---

##  Future Improvements

*  Deep Learning Integration
*  Real ECG Signal Processing
*  Mobile App Integration
*  Live Sensor Data Streaming



