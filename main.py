from fastapi import FastAPI, WebSocket
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("../outputs/model/emotion_model.pkl")
scaler = joblib.load("../outputs/model/scaler.pkl")

# -------------------------------
# NORMAL API
# -------------------------------
@app.get("/")
def home():
    return {"message": "AI Health System Running"}

# -------------------------------
# PREDICTION API
# -------------------------------
@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["hr"],
        data["rmssd"],
        data["sdnn"],
        data["lf_hf"],
        data["pnn50"]
    ]])

    features = scaler.transform(features)
    pred = model.predict(features)[0]

    mapping = {
        0:"CALM 😊",
        1:"NORMAL 🙂",
        2:"STRESSED 😡",
        3:"ANXIOUS 😰"
    }

    return {"emotion": mapping[pred]}

# -------------------------------
# WEBSOCKET (REAL-TIME)
# -------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()

        features = np.array([[
            data["hr"],
            data["rmssd"],
            data["sdnn"],
            data["lf_hf"],
            data["pnn50"]
        ]])

        features = scaler.transform(features)
        pred = model.predict(features)[0]

        mapping = {
            0:"CALM 😊",
            1:"NORMAL 🙂",
            2:"STRESSED 😡",
            3:"ANXIOUS 😰"
        }

        await websocket.send_json({
            "emotion": mapping[pred]
        })