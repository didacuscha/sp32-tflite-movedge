"""
Convierte el modelo Keras entrenado (train_tf_model.py) a TensorFlow Lite,
en dos variantes, y compara tamano/latencia/precision entre ellas:

  1. float32   -- sin cuantizar, referencia de precision maxima.
  2. int8 full -- cuantizacion post-entrenamiento full-integer (pesos y
                  activaciones en int8), usando un representative_dataset
                  para calibrar los rangos. Esta es la variante que se
                  embebe en el firmware de la ESP32 (menor tamano/RAM,
                  requerido por TFLite Micro para el mejor rendimiento).

Guarda ambos .tflite en models/ y un reporte de comparacion en
models/tflite_comparison.json.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.csv")
KERAS_PATH = os.path.join(MODEL_DIR, "movement_classifier.keras")
FEATURES = ["Accel_x", "Accel_y", "Accel_z", "w_x", "w_y", "w_z"]
CLASSES = ["Mov00", "Mov01", "Mov02", "Mov03", "Mov04"]


def load_test_split():
    """Recrea el mismo split y escalado usados en train_tf_model.py para
    evaluar en datos que el modelo no vio durante el entrenamiento, en el
    mismo dominio (ya escalado) que el modelo espera."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype("float32")
    y = np.array([CLASSES.index(c) for c in df["Category"].values], dtype="int64")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    with open(os.path.join(MODEL_DIR, "scaler.json")) as f:
        scaler = json.load(f)
    mean = np.array(scaler["mean"], dtype="float32")
    scale = np.array(scaler["scale"], dtype="float32")
    X_test_s = (X_test - mean) / scale
    return X_test_s.astype("float32"), y_test


def convert_float(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_int8(model, representative_samples):
    def representative_dataset():
        for i in range(representative_samples.shape[0]):
            yield [representative_samples[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32  # firmware feeds raw sensor floats
    converter.inference_output_type = tf.float32
    return converter.convert()


def evaluate_tflite(tflite_bytes, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = np.zeros(len(X_test), dtype="int64")
    start = time.perf_counter()
    for i in range(len(X_test)):
        interpreter.set_tensor(input_details["index"], X_test[i : i + 1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        preds[i] = int(np.argmax(out[0]))
    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    avg_latency_us = (elapsed / len(X_test)) * 1e6
    return acc, f1, avg_latency_us


def main():
    model = tf.keras.models.load_model(KERAS_PATH)
    X_test, y_test = load_test_split()
    representative_samples = np.load(
        os.path.join(MODEL_DIR, "representative_samples.npy")
    )

    print("Convirtiendo a TensorFlow Lite (float32)...")
    tflite_float = convert_float(model)
    float_path = os.path.join(MODEL_DIR, "movement_classifier_float.tflite")
    with open(float_path, "wb") as f:
        f.write(tflite_float)

    print("Convirtiendo a TensorFlow Lite (int8, full-integer quantization)...")
    tflite_int8 = convert_int8(model, representative_samples)
    int8_path = os.path.join(MODEL_DIR, "movement_classifier_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)

    print("\nEvaluando float32 .tflite en el test set...")
    float_acc, float_f1, float_latency = evaluate_tflite(tflite_float, X_test, y_test)

    print("Evaluando int8 .tflite en el test set...")
    int8_acc, int8_f1, int8_latency = evaluate_tflite(tflite_int8, X_test, y_test)

    comparison = {
        "float32": {
            "path": os.path.basename(float_path),
            "size_bytes": len(tflite_float),
            "test_accuracy": float(float_acc),
            "test_macro_f1": float(float_f1),
            "avg_inference_latency_us_host": float(float_latency),
        },
        "int8": {
            "path": os.path.basename(int8_path),
            "size_bytes": len(tflite_int8),
            "test_accuracy": float(int8_acc),
            "test_macro_f1": float(int8_f1),
            "avg_inference_latency_us_host": float(int8_latency),
        },
    }
    comparison["size_reduction_pct"] = 100.0 * (
        1 - comparison["int8"]["size_bytes"] / comparison["float32"]["size_bytes"]
    )

    print("\n" + "=" * 70)
    print("COMPARACION float32 vs int8")
    print("=" * 70)
    print(json.dumps(comparison, indent=2))

    with open(os.path.join(MODEL_DIR, "tflite_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nReporte guardado en {os.path.join(MODEL_DIR, 'tflite_comparison.json')}")
    print(
        "\nNota: la latencia medida aqui es en el host (interprete Python de "
        "TFLite), solo sirve como referencia relativa entre variantes -- la "
        "latencia real en la ESP32 se mide en el firmware (TFLite Micro)."
    )


if __name__ == "__main__":
    main()
