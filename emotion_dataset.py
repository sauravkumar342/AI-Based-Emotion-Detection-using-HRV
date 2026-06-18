import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------ CREATE FOLDERS ------------------
os.makedirs("../outputs/graphs", exist_ok=True)
os.makedirs("../outputs/reports", exist_ok=True)
os.makedirs("../outputs/model", exist_ok=True)

# ------------------ GENERATE DATA ------------------
data = []
n = 5000

def gen(hr_m, hr_s, rm_m, rm_s, sd_m, sd_s, label):
    for _ in range(n):
        hr = int(np.random.normal(hr_m, hr_s))
        rmssd = int(np.random.normal(rm_m, rm_s))
        sdnn = int(np.random.normal(sd_m, sd_s))
        lf_hf = np.random.uniform(0.5, 3.5)
        pnn50 = np.random.uniform(5, 60)

        hr = max(50, min(hr, 130))
        rmssd = max(5, min(rmssd, 100))
        sdnn = max(10, min(sdnn, 120))

        data.append([hr, rmssd, sdnn, lf_hf, pnn50, label])

gen(65,8,60,10,70,15,"CALM")
gen(80,10,40,10,50,15,"NORMAL")
gen(95,10,25,8,30,10,"STRESSED")
gen(110,10,15,5,20,8,"ANXIOUS")

df = pd.DataFrame(data, columns=["hr","rmssd","sdnn","lf_hf","pnn50","emotion"])

# ------------------ GRAPHS ------------------
sns.histplot(df["hr"], kde=True)
plt.savefig("../outputs/graphs/distribution.png")
plt.clf()

sns.pairplot(df.sample(1000), hue="emotion")
plt.savefig("../outputs/graphs/pairplot.png")
plt.close()

# ------------------ PREPROCESS ------------------
X = df[["hr","rmssd","sdnn","lf_hf","pnn50"]]
y = df["emotion"].map({"CALM":0,"NORMAL":1,"STRESSED":2,"ANXIOUS":3})

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# ------------------ MODEL ------------------
model = ExtraTreesClassifier()

params = {
    "n_estimators":[200,300],
    "max_depth":[None,20]
}

grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train,y_train)

best_model = grid.best_estimator_

# ------------------ PREDICTION ------------------
pred = best_model.predict(X_test)
acc = accuracy_score(y_test, pred)

# ------------------ SAVE TABLE ------------------
pd.DataFrame({
    "Model":["ExtraTrees"],
    "Accuracy":[acc]
}).to_csv("../outputs/reports/accuracy.csv", index=False)

# ------------------ CONFUSION ------------------
cm = confusion_matrix(y_test, pred)

sns.heatmap(cm, annot=True)
plt.savefig("../outputs/graphs/confusion.png")
plt.clf()

# ------------------ ROC ------------------
probs = best_model.predict_proba(X_test)

for i in range(4):
    fpr, tpr, _ = roc_curve((y_test==i), probs[:,i])
    plt.plot(fpr, tpr)

plt.savefig("../outputs/graphs/roc.png")
plt.clf()

# ------------------ FEATURE IMPORTANCE ------------------
imp = best_model.feature_importances_
features = ["hr","rmssd","sdnn","lf_hf","pnn50"]

plt.bar(features, imp)
plt.savefig("../outputs/graphs/importance.png")
plt.clf()

# ------------------ REPORT ------------------
report = classification_report(y_test, pred)

with open("../outputs/reports/report.txt","w") as f:
    f.write(report)

# ------------------ SAVE MODEL ------------------
joblib.dump(best_model, "outputs/model/emotion_model.pkl")
joblib.dump(scaler, "outputs/model/scaler.pkl")

print("✅ FULL AI SYSTEM DONE")