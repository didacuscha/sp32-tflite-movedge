"""
Entrena el clasificador de movimientos usando TensorFlow (Keras).

A diferencia de train_classifier_reference.py (regresion logistica manual
via sklearn/statsmodels, sin TensorFlow, del proyecto movedge), este script
construye un modelo Keras real (MLP pequeno) y se convierte despues a
TensorFlow Lite en convert_to_tflite.py.

Preprocesamiento -- IMPORTANTE: el escalado (StandardScaler) se aplica
en Python ANTES de que los datos entren al modelo, en vez de meterlo como
una capa Normalization dentro del grafo. Se probo la version con
Normalization-en-el-grafo primero y rompia la cuantizacion int8: Accel_*
tiene rango ~[-10, 10] m/s^2 mientras que w_* (giroscopio) tiene rango
~[-0.1, 0.1] rad/s (~100x mas chico). La cuantizacion int8 de TFLite usa
UNA sola escala por tensor de entrada, calibrada con el rango combinado de
las 6 features -- con features en rangos tan distintos, esa escala queda
dominada por Accel y los ejes de giroscopio colapsan a 2-3 niveles
distinguibles, destruyendo la senal que el modelo necesita para distinguir
movimientos direccionales (test accuracy caia a ~55%). Escalando ANTES de
la cuantizacion, las 6 features quedan en un rango comparable y la
cuantizacion int8 funciona correctamente (ver convert_to_tflite.py).

El firmware debe replicar este mismo escalado (mean/scale, guardados en
models/scaler.json) antes de invocar el interprete de TFLite Micro.

Dataset reutilizado del proyecto anterior (movedge): data/training_data.csv.
Temp se excluye como feature por la misma razon documentada en
train_classifier_reference.py: en esa sesion de captura la temperatura
sube monotonicamente por autocalentamiento del sensor a medida que pasa el
tiempo, quedando correlacionada con el ORDEN de captura y no con el
movimiento en si.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURES = ["Accel_x", "Accel_y", "Accel_z", "w_x", "w_y", "w_z"]
CLASSES = ["Mov00", "Mov01", "Mov02", "Mov03", "Mov04"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype("float32")
    y = np.array([CLASSES.index(c) for c in df["Category"].values], dtype="int64")
    return X, y


def build_model():
    inputs = tf.keras.Input(shape=(len(FEATURES),), name="features")
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="movement_classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype("float32")
    X_test_s = scaler.transform(X_test).astype("float32")

    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model.fit(
        X_train_s,
        y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    y_prob = model.predict(X_test_s, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nTest accuracy: {acc:.4f}  |  Macro-F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=real, cols=predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=CLASSES, columns=CLASSES))

    os.makedirs(MODEL_DIR, exist_ok=True)
    keras_path = os.path.join(MODEL_DIR, "movement_classifier.keras")
    model.save(keras_path)
    print(f"\nModelo Keras guardado en {keras_path}")

    # Muestras YA ESCALADAS (mismo dominio que vera el modelo en inferencia)
    # para usar como representative_dataset en la cuantizacion int8
    # (convert_to_tflite.py).
    np.save(os.path.join(MODEL_DIR, "representative_samples.npy"), X_train_s[:500])

    scaler_params = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(os.path.join(MODEL_DIR, "scaler.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Parametros del scaler guardados en {os.path.join(MODEL_DIR, 'scaler.json')}")

    metrics = {
        "test_accuracy": float(acc),
        "test_macro_f1": float(f1),
        "features": FEATURES,
        "classes": CLASSES,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    with open(os.path.join(MODEL_DIR, "tf_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metricas guardadas en {os.path.join(MODEL_DIR, 'tf_model_metrics.json')}")


if __name__ == "__main__":
    main()
