"""
Captura datos etiquetados del acelerometro/giroscopio de la ESP32 (MPU6050)
por el puerto serial y los guarda en data.csv.

Requisito previo: firmware de firmware/src/main.cpp ya flasheado en la ESP32
(via `pio run -t upload` desde firmware/), e imprimiendo lineas
"accel_x;accel_y;accel_z;gyro_x;gyro_y;gyro_z;temp".

Cierre cualquier monitor serial (PlatformIO, Arduino IDE, screen, etc.)
antes de correr este script -- el puerto serial solo lo puede tener
abierto un proceso a la vez.
"""

import os
import serial
import time
import pandas as pd
import numpy as np

SERIAL_PORT = "/dev/cu.usbserial-110"
BAUDRATE = 115200
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

COL_NAMES = ["Accel_x", "Accel_y", "Accel_z", "w_x", "w_y", "w_z", "Temp"]


class Esp32Communication:
    def __init__(self, port=SERIAL_PORT, baudrate=BAUDRATE, timeout=0.1):
        self.esp32 = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def read_port(self):
        packet = self.esp32.readline()
        return packet.decode("utf-8", errors="ignore").rstrip("\n")

    def close(self):
        self.esp32.close()


def parse_lines(raw_lines):
    """Convierte lineas 'a;b;c;d;e;f;g' en un DataFrame de 7 columnas."""
    rows = []
    for line in raw_lines:
        parts = line.split(";")
        if len(parts) != 7:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=COL_NAMES)


def capture_data(category, duration=15):
    esp32_comm = Esp32Communication()
    data_list = []
    try:
        esp32_comm.esp32.reset_input_buffer()
        start_time = time.time()
        print(f"Capturando datos para la categoria '{category}' durante {duration} segundos...")
        while (time.time() - start_time) < duration:
            value = esp32_comm.read_port()
            if value:
                data_list.append(value)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        esp32_comm.close()

    df = parse_lines(data_list)
    df["Category"] = category
    print(f"  -> {len(df)} muestras capturadas.")
    return df


if __name__ == "__main__":
    print("Bienvenido a la captura de datos de tus movimientos")
    data_frames = []

    # Rutina controlada por tiempo (protocolo de la guia): cada movimiento
    # 15 segundos, en este orden. Mov00 se repite entre cada gesto para
    # tener una linea base limpia y mas muestras de "quieto".
    routine = [
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

    for i, (category, duration) in enumerate(routine, start=1):
        print(f"[{i}/{len(routine)}] Preparandote para capturar el movimiento '{category}'")
        input("Presiona Enter para comenzar...")
        df = capture_data(category, duration)
        data_frames.append(df)

    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_filename = os.path.join(OUTPUT_DIR, "training_data_new.csv")
        final_df.to_csv(csv_filename, index=False)
        print(f"Datos guardados en {csv_filename}")
        print(final_df["Category"].value_counts())
    else:
        print("No se capturaron datos.")
