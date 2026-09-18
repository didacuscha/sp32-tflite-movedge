"""
Protocolo de pruebas final (fase 6 de la guia). Requiere que el firmware de
src/main.cpp YA este flasheado y corriendo en la ESP32 (clasificador en
tiempo real vía FreeRTOS) -- este script NO reflashea, solo abre el puerto
serial, orquesta los tiempos/movimientos, y junta cada linea que imprime el
firmware (features + Clase Predecida + probabilidades) con la Clase Real
(el movimiento que el protocolo le pidio hacer en ese instante).

Rutina (exactamente como la pide la guia):
  1) Estructurada, 15s cada uno, en este orden:
     Mov00, Mov01, Mov00, Mov02, Mov00, Mov03, Mov00, Mov04, Mov00
  2) Secuencia aleatoria de movimientos (incluye Mov00), en bloques de 5s,
     durante 30s en total (6 bloques).

Salida: test_protocol_results.csv con las columnas exactas de la tabla de
la guia (Feature1..Feature7, Clase Real, Clase Predecida, Prob Mov00..04).
Para la entrega final, este archivo se renombra a data.csv (ver mensaje
final del script) -- se deja con otro nombre aqui para no pisar el
data.csv de entrenamiento que ya tenemos.
"""

import random
import time
import serial
import pandas as pd

SERIAL_PORT = "/dev/cu.usbserial-110"
BAUDRATE = 115200

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


def build_random_routine():
    return [(random.choice(CLASSES), RANDOM_SEGMENT_DURATION) for _ in range(N_RANDOM_SEGMENTS)]


def parse_firmware_line(line):
    """Linea esperada (impresa por TaskVisualizacion en main.cpp):
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
        for cls, p in zip(CLASSES, probs):
            row[f"Prob {cls}"] = p
        rows.append(row)
        n_rows += 1
    print(f"     {n_rows} muestras registradas.")


def main():
    random.seed()  # aleatoriedad real en cada corrida
    random_routine = build_random_routine()
    full_routine = STRUCTURED_ROUTINE + random_routine

    print("=" * 70)
    print("PROTOCOLO DE PRUEBAS - RTOS-MovEdge (fase 6)")
    print("=" * 70)
    print("Asegurese de que el firmware (main.cpp) ya esta flasheado y")
    print("corriendo en la ESP32, y de que no haya otro monitor serial abierto.")
    print()
    print("Rutina estructurada (15s cada uno):")
    print("  " + " -> ".join(l for l, _ in STRUCTURED_ROUTINE))
    print(f"Rutina aleatoria (bloques de {RANDOM_SEGMENT_DURATION}s, {RANDOM_TOTAL_DURATION}s en total):")
    print("  " + " -> ".join(l for l, _ in random_routine))
    print()
    input("Presione Enter para abrir el puerto serial y comenzar...")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # dar tiempo a que la ESP32 termine de reiniciar tras abrir el puerto
    ser.reset_input_buffer()

    rows = []
    for i, (label, duration) in enumerate(full_routine, start=1):
        print(f"\n[{i}/{len(full_routine)}] Siguiente movimiento: {label}")
        input("Colocate en posicion y presiona Enter para comenzar el bloque...")
        run_segment(ser, label, duration, rows)

    ser.close()

    df = pd.DataFrame(rows, columns=FEATURE_COLS + ["Clase Real", "Clase Predecida"] +
                       [f"Prob {c}" for c in CLASSES])
    out_file = "test_protocol_results.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de muestras registradas: {len(df)}")
    accuracy = (df["Clase Real"] == df["Clase Predecida"]).mean()
    print(f"Accuracy global del protocolo: {accuracy:.4f}")
    print("\nMatriz de confusion (filas=Real, columnas=Predecida):")
    print(pd.crosstab(df["Clase Real"], df["Clase Predecida"]))
    print(f"\nResultados guardados en {out_file}")
    print("Para la entrega final, copie/renombre este archivo a 'data.csv'")
    print("(no se sobreescribe automaticamente el data.csv de entrenamiento).")


if __name__ == "__main__":
    main()
