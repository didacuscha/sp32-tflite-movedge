"""
Genera los headers C que el firmware necesita para la inferencia on-device
(step 8), a partir de los artefactos ya generados por train_tf_model.py y
convert_to_tflite.py:

  - firmware/include/model_data.h      <- models/movement_classifier_int8.tflite
                                           empacado como arreglo de bytes.
  - firmware/include/movement_scaler.h <- models/scaler.json (mean/scale por
                                           feature) + la lista de clases.

El modelo se embebe tal cual (ya incluye quantize/dequantize internos
porque se convirtio con inference_input_type/output_type=float32), asi que
el firmware solo necesita escalar las features (igual que en Python) antes
de escribirlas en el tensor de entrada.
"""

import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIRMWARE_INCLUDE_DIR = os.path.join(os.path.dirname(__file__), "..", "firmware", "include")
TFLITE_PATH = os.path.join(MODEL_DIR, "movement_classifier_int8.tflite")


def export_model_header():
    with open(TFLITE_PATH, "rb") as f:
        model_bytes = f.read()

    lines = [
        "// AUTO-GENERADO por training/export_firmware_headers.py -- no editar a mano.",
        f"// Fuente: models/{os.path.basename(TFLITE_PATH)} ({len(model_bytes)} bytes)",
        "#pragma once",
        "",
        f"alignas(16) const unsigned char g_movement_model[] = {{",
    ]
    hex_bytes = [f"0x{b:02x}" for b in model_bytes]
    for i in range(0, len(hex_bytes), 12):
        lines.append("    " + ", ".join(hex_bytes[i : i + 12]) + ",")
    lines.append("};")
    lines.append(f"const unsigned int g_movement_model_len = {len(model_bytes)};")
    lines.append("")

    out_path = os.path.join(FIRMWARE_INCLUDE_DIR, "model_data.h")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Escrito {out_path} ({len(model_bytes)} bytes empacados)")


def export_scaler_header():
    with open(os.path.join(MODEL_DIR, "scaler.json")) as f:
        scaler = json.load(f)
    with open(os.path.join(MODEL_DIR, "tf_model_metrics.json")) as f:
        metrics = json.load(f)

    features = scaler["features"]
    classes = metrics["classes"]
    mean = scaler["mean"]
    scale = scaler["scale"]

    def carr(values):
        return ", ".join(f"{v:.8f}f" for v in values)

    lines = [
        "// AUTO-GENERADO por training/export_firmware_headers.py -- no editar a mano.",
        f"// Escalado (StandardScaler) entrenado en train_tf_model.py sobre {metrics['train_samples']} muestras.",
        "#pragma once",
        "",
        f"#define MC_N_FEATURES {len(features)}",
        f"#define MC_N_CLASSES {len(classes)}",
        "",
        "static const char* MC_CLASS_NAMES[MC_N_CLASSES] = {",
        "    " + ", ".join(f'"{c}"' for c in classes),
        "};",
        "",
        "// Orden de features: " + ", ".join(features),
        f"static const float MC_SCALER_MEAN[MC_N_FEATURES] = {{ {carr(mean)} }};",
        f"static const float MC_SCALER_SCALE[MC_N_FEATURES] = {{ {carr(scale)} }};",
        "",
        "// Escala las features crudas al mismo dominio visto en entrenamiento",
        "// (ver train_tf_model.py). El modelo TFLite ya incluye internamente",
        "// la cuantizacion/decuantizacion int8, asi que aqui solo se replica",
        "// el StandardScaler de Python.",
        "static inline void mc_scale_features(const float raw[MC_N_FEATURES], float scaled_out[MC_N_FEATURES]) {",
        "    for (int i = 0; i < MC_N_FEATURES; i++) {",
        "        scaled_out[i] = (raw[i] - MC_SCALER_MEAN[i]) / MC_SCALER_SCALE[i];",
        "    }",
        "}",
        "",
    ]

    out_path = os.path.join(FIRMWARE_INCLUDE_DIR, "movement_scaler.h")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Escrito {out_path}")


def main():
    os.makedirs(FIRMWARE_INCLUDE_DIR, exist_ok=True)
    export_model_header()
    export_scaler_header()


if __name__ == "__main__":
    main()
