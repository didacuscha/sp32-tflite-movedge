// AUTO-GENERADO por training/export_firmware_headers.py -- no editar a mano.
// Escalado (StandardScaler) entrenado en train_tf_model.py sobre 7945 muestras.
#pragma once

#define MC_N_FEATURES 6
#define MC_N_CLASSES 5

static const char* MC_CLASS_NAMES[MC_N_CLASSES] = {
    "Mov00", "Mov01", "Mov02", "Mov03", "Mov04"
};

// Orden de features: Accel_x, Accel_y, Accel_z, w_x, w_y, w_z
static const float MC_SCALER_MEAN[MC_N_FEATURES] = { -0.22337570f, 0.03351415f, 4.71773445f, -0.00170422f, 0.00078288f, 0.03748269f };
static const float MC_SCALER_SCALE[MC_N_FEATURES] = { 4.60336798f, 4.62806252f, 4.65263388f, 0.00797514f, 0.01016621f, 0.00854893f };

// Escala las features crudas al mismo dominio visto en entrenamiento
// (ver train_tf_model.py). El modelo TFLite ya incluye internamente
// la cuantizacion/decuantizacion int8, asi que aqui solo se replica
// el StandardScaler de Python.
static inline void mc_scale_features(const float raw[MC_N_FEATURES], float scaled_out[MC_N_FEATURES]) {
    for (int i = 0; i < MC_N_FEATURES; i++) {
        scaled_out[i] = (raw[i] - MC_SCALER_MEAN[i]) / MC_SCALER_SCALE[i];
    }
}
