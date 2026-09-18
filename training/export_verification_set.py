"""
Genera cpp_verification_set.csv: filas reales de data.csv junto con la
probabilidad/clase que el modelo (a partir de model_params.json) predice en
Python (numpy puro), para comparar bit-a-bit contra la implementacion en C++
antes de subirla a la ESP32.
"""

import json
import numpy as np
import pandas as pd

FEATURES = ["Accel_x", "Accel_y", "Accel_z", "w_x", "w_y", "w_z"]  # sin Temp, ver train_classifier.py
CLASSES = ["Mov00", "Mov01", "Mov02", "Mov03", "Mov04"]

with open("model_params.json") as f:
    m = json.load(f)

mean = np.array(m["scaler_mean"])
scale = np.array(m["scaler_scale"])
coef = np.array(m["coefficients"])       # (5,7)
intercept = np.array(m["intercepts"])    # (5,)

df = pd.read_csv("training_data.csv")
sample = df.groupby("Category", group_keys=False).sample(n=10, random_state=42).reset_index(drop=True)

X = sample[FEATURES].values
y_true = sample["Category"].values

Xs = (X - mean) / scale
logits = Xs @ coef.T + intercept          # (N,5)
logits -= logits.max(axis=1, keepdims=True)
exp = np.exp(logits)
probs = exp / exp.sum(axis=1, keepdims=True)
pred_idx = probs.argmax(axis=1)

out = sample[FEATURES].copy()
out["true_label"] = y_true
out["pred_label_py"] = [CLASSES[i] for i in pred_idx]
for i, c in enumerate(CLASSES):
    out[f"prob_{c}_py"] = probs[:, i]

out.to_csv("cpp_verification_set.csv", index=False)
print(f"Exportadas {len(out)} filas a cpp_verification_set.csv")
print(out[["true_label", "pred_label_py"]].value_counts())
