// AUTO-GENERADO por generate_classifier_header.py -- no editar a mano.
// Modelo: logistic_regression (softmax) | test_accuracy=1.0000 test_macro_f1=1.0000
#pragma once

#include <math.h>

#define MC_N_CLASSES 5
#define MC_N_FEATURES 6

static const char* MC_CLASS_NAMES[MC_N_CLASSES] = {
    "Mov00", "Mov01", "Mov02", "Mov03", "Mov04"
};

// Orden de features: Accel_x, Accel_y, Accel_z, w_x, w_y, w_z
static const float MC_SCALER_MEAN[MC_N_FEATURES] = { -0.22337571f, 0.03351416f, 4.71773442f, -0.00170422f, 0.00078288f, 0.03748269f };
static const float MC_SCALER_SCALE[MC_N_FEATURES] = { 4.60336799f, 4.62806251f, 4.65263385f, 0.00797514f, 0.01016621f, 0.00854893f };

static const float MC_COEF[MC_N_CLASSES][MC_N_FEATURES] = {
    { -0.17540993f, -0.05423464f, 2.01015336f, 0.11677368f, -0.03247126f, 0.11760971f },
    { 0.05317503f, -1.67740233f, -0.41885589f, 0.02124327f, 0.10755773f, -0.10264179f },
    { 0.02141723f, 1.73143946f, -0.38269262f, -0.02710333f, 0.02082674f, -0.05787478f },
    { -1.48816532f, -0.00507702f, -0.74255184f, -0.02654484f, -0.02216357f, 0.01190448f },
    { 1.58898298f, 0.00527453f, -0.46605301f, -0.08436878f, -0.07374964f, 0.03100239f },
};

static const float MC_INTERCEPT[MC_N_CLASSES] = { 2.24252675f, -0.55252790f, -0.47842548f, -0.61148527f, -0.60008810f };


// Estandariza, calcula logits (regresion logistica multinomial) y aplica
// softmax. Escribe las 5 probabilidades en probs_out (suman 1) y retorna
// el indice de la clase con mayor probabilidad.
static inline int mc_classify(const float features[MC_N_FEATURES], float probs_out[MC_N_CLASSES]) {
    float scaled[MC_N_FEATURES];
    for (int i = 0; i < MC_N_FEATURES; i++) {
        scaled[i] = (features[i] - MC_SCALER_MEAN[i]) / MC_SCALER_SCALE[i];
    }

    float logits[MC_N_CLASSES];
    float max_logit = -INFINITY;
    for (int c = 0; c < MC_N_CLASSES; c++) {
        float z = MC_INTERCEPT[c];
        for (int i = 0; i < MC_N_FEATURES; i++) {
            z += MC_COEF[c][i] * scaled[i];
        }
        logits[c] = z;
        if (z > max_logit) max_logit = z;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < MC_N_CLASSES; c++) {
        probs_out[c] = expf(logits[c] - max_logit);
        sum_exp += probs_out[c];
    }

    int best = 0;
    for (int c = 0; c < MC_N_CLASSES; c++) {
        probs_out[c] /= sum_exp;
        if (probs_out[c] > probs_out[best]) best = c;
    }
    return best;
}
