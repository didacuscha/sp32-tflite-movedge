"""
Genera include/movement_classifier.h a partir de model_params.json.
Evita transcribir los coeficientes a mano (fuente de errores).
"""

import json

with open("model_params.json") as f:
    m = json.load(f)

classes = m["classes"]
n_classes = len(classes)
n_features = len(m["features"])
mean = m["scaler_mean"]
scale = m["scaler_scale"]
coef = m["coefficients"]
intercept = m["intercepts"]


def carr(values, fmt="{:.8f}f"):
    return ", ".join(fmt.format(v) for v in values)


lines = []
lines.append("// AUTO-GENERADO por generate_classifier_header.py -- no editar a mano.")
lines.append(f"// Modelo: {m['model_type']} ({m['link_function']}) | "
             f"test_accuracy={m['test_accuracy']:.4f} test_macro_f1={m['test_macro_f1']:.4f}")
lines.append("#pragma once")
lines.append("")
lines.append("#include <math.h>")
lines.append("")
lines.append(f"#define MC_N_CLASSES {n_classes}")
lines.append(f"#define MC_N_FEATURES {n_features}")
lines.append("")
lines.append("static const char* MC_CLASS_NAMES[MC_N_CLASSES] = {")
lines.append("    " + ", ".join(f'"{c}"' for c in classes))
lines.append("};")
lines.append("")
lines.append("// Orden de features: " + ", ".join(m["features"]))
lines.append(f"static const float MC_SCALER_MEAN[MC_N_FEATURES] = {{ {carr(mean)} }};")
lines.append(f"static const float MC_SCALER_SCALE[MC_N_FEATURES] = {{ {carr(scale)} }};")
lines.append("")
lines.append("static const float MC_COEF[MC_N_CLASSES][MC_N_FEATURES] = {")
for row in coef:
    lines.append("    { " + carr(row) + " },")
lines.append("};")
lines.append("")
lines.append(f"static const float MC_INTERCEPT[MC_N_CLASSES] = {{ {carr(intercept)} }};")
lines.append("")
lines.append("""
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
""")

with open("include/movement_classifier.h", "w") as f:
    f.write("\n".join(lines))

print("Escrito include/movement_classifier.h")
