"""
Protocolo de pruebas final (entregable de la guia). Requiere que el firmware
de firmware/src/main.cpp YA este flasheado y corriendo en la ESP32
(clasificador TFLite Micro en tiempo real via FreeRTOS, ver
firmware/src/main.cpp) -- este script NO reflashea, solo abre el puerto
serial, orquesta los tiempos/movimientos, y junta cada linea que imprime el
firmware (features + Clase Predecida + probabilidades) con la Clase Real
(el movimiento que el protocolo pidio hacer en ese instante).

IMPORTANTE sobre las poses: estas "clases" de movimiento son POSES
ESTATICAS (el acelerometro redirige el vector de gravedad segun la
inclinacion; el giroscopio se mantiene ~0 en todas las clases), no gestos
dinamicos. Ver data/training_data.csv / README para las orientaciones
objetivo por clase.

Rutina (exactamente como la pide la guia):
  1) Estructurada, 15s cada uno, en este orden:
     Mov00, Mov01, Mov00, Mov02, Mov00, Mov03, Mov00, Mov04, Mov00
  2) Secuencia aleatoria de movimientos (incluye Mov00), en bloques de 5s,
     durante 30s en total (6 bloques).

Salida (en protocol/results/):
  - protocol_run_full.csv : Feature1..7, Clase Real, Clase Predecida,
    Resultado Inferencia, y ademas Prob Mov00..04 (detalle completo, para
    el reporte).
  - data.csv : subconjunto con EXACTAMENTE las columnas de la tabla de la
    guia (Feature1..7, Clase Real, Clase Predecida, Resultado Inferencia),
    listo para el entregable final.

Nota sobre "Resultado Inferencia": la guia lo describe como "resultado de
la inferencia (% reconocimiento)" sin una formula explicita. Se define
aqui como la probabilidad (softmax) que el modelo asigno a la clase que
PREDIJO (max(probs)) -- la lectura mas estandar de "% de reconocimiento"
para un resultado de clasificacion. Documentado tambien en el reporte.

Modos de uso:
  - Interactivo (uso real, en el laboratorio): python run_test_protocol.py
    Espera Enter antes de cada bloque para que puedas acomodar la pose.
  - Automatico (--auto): no espera Enter, corre los tiempos exactos
    seguidos. Util cuando alguien mas anuncia los tiempos en voz alta o
    por chat en vez de leerlos en la terminal.
"""

import argparse
import os
import random
import time

import pandas as pd
import serial

SERIAL_PORT = "/dev/cu.usbserial-110"
BAUDRATE = 115200
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

CLASSES = ["Mov00", "Mov01", "Mov02", "Mov03", "Mov04"]
FEATURE_COLS = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7"]

STRUCTURED_ROUTINE = [
    ("Mov00", 15),
    ("Mov01", 15),
    ("Mov00", 15),
    ("Mov02", 15),
    ("Mov00", 15),
    ("Mov03", 15),
    ("Mov00", 15),
    ("Mov04", 15),
    ("Mov00", 15),
]

RANDOM_SEGMENT_DURATION = 5
RANDOM_TOTAL_DURATION = 30
N_RANDOM_SEGMENTS = RANDOM_TOTAL_DURATION // RANDOM_SEGMENT_DURATION


def build_random_routine(seed=None):
    rng = random.Random(seed)
    return [(rng.choice(CLASSES), RANDOM_SEGMENT_DURATION) for _ in range(N_RANDOM_SEGMENTS)]


def parse_firmware_line(line):
    """Linea esperada (impresa por vTaskVisualizacion en firmware/src/main.cpp):
    Accel_x;Accel_y;Accel_z;w_x;w_y;w_z;Temp;ClasePredecida;ProbMov00;...;ProbMov04
    Retorna None si la linea no tiene el formato esperado (p.ej. el encabezado)."""
    parts = line.split(";")
    if len(parts) != 13:
        return None
    try:
        features = [float(p) for p in parts[:7]]
        pred_label = parts[7]
        probs = [float(p) for p in parts[8:13]]
    except ValueError:
        return None
    if pred_label not in CLASSES:
        return None
    return features, pred_label, probs


def run_segment(ser, real_label, duration, rows):
    print(f"  -> Ejecutando '{real_label}' durante {duration}s...")
    start = time.time()
    n_rows = 0
    while (time.time() - start) < duration:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw:
            continue
        parsed = parse_firmware_line(raw)
        if parsed is None:
            continue
        features, pred_label, probs = parsed
        row = dict(zip(FEATURE_COLS, features))
        row["Clase Real"] = real_label
        row["Clase Predecida"] = pred_label
        row["Resultado Inferencia"] = max(probs)
        for cls, p in zip(CLASSES, probs):
            row[f"Prob {cls}"] = p
        rows.append(row)
        n_rows += 1
    print(f"     {n_rows} muestras registradas.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                         help="No esperar Enter entre bloques; correr los tiempos seguidos.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Semilla para la secuencia aleatoria (reproducibilidad).")
    args = parser.parse_args()

    random_routine = build_random_routine(args.seed)
    full_routine = STRUCTURED_ROUTINE + random_routine

    print("=" * 70)
    print("PROTOCOLO DE PRUEBAS - sp32-tflite-movedge")
    print("=" * 70)
    print("Asegurese de que el firmware (firmware/src/main.cpp) ya esta")
    print("flasheado y corriendo en la ESP32, y de que no haya otro monitor")
    print("serial abierto.")
    print()
    print("Rutina estructurada (15s cada uno):")
    print("  " + " -> ".join(l for l, _ in STRUCTURED_ROUTINE))
    print(f"Rutina aleatoria (bloques de {RANDOM_SEGMENT_DURATION}s, {RANDOM_TOTAL_DURATION}s en total):")
    print("  " + " -> ".join(l for l, _ in random_routine))
    print()
    if not args.auto:
        input("Presione Enter para abrir el puerto serial y comenzar...")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # dar tiempo a que la ESP32 termine de reiniciar tras abrir el puerto
    ser.reset_input_buffer()

    rows = []
    for i, (label, duration) in enumerate(full_routine, start=1):
        print(f"\n[{i}/{len(full_routine)}] Siguiente movimiento: {label}")
        if not args.auto:
            input("Colocate en posicion y presiona Enter para comenzar el bloque...")
        run_segment(ser, label, duration, rows)

    ser.close()

    all_cols = FEATURE_COLS + ["Clase Real", "Clase Predecida", "Resultado Inferencia"] + \
        [f"Prob {c}" for c in CLASSES]
    df = pd.DataFrame(rows, columns=all_cols)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_path = os.path.join(RESULTS_DIR, "protocol_run_full.csv")
    df.to_csv(full_path, index=False)

    deliverable_cols = FEATURE_COLS + ["Clase Real", "Clase Predecida", "Resultado Inferencia"]
    deliverable_path = os.path.join(RESULTS_DIR, "data.csv")
    df[deliverable_cols].to_csv(deliverable_path, index=False)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de muestras registradas: {len(df)}")
    accuracy = (df["Clase Real"] == df["Clase Predecida"]).mean()
    print(f"Accuracy global del protocolo: {accuracy:.4f}")
    print("\nMatriz de confusion (filas=Real, columnas=Predecida):")
    print(pd.crosstab(df["Clase Real"], df["Clase Predecida"]))
    print(f"\nDetalle completo guardado en {full_path}")
    print(f"Entregable (formato exacto de la guia) guardado en {deliverable_path}")


if __name__ == "__main__":
    main()
