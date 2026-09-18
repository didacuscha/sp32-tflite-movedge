"""
Entrena el clasificador de movimientos (Probit y Regresion Logistica) sobre
data.csv, y exporta los parametros del mejor modelo a model_params.json para
despues hardcodearlos en el firmware de la ESP32 (fase 4).

Restriccion de hardware (ESP32, sin librerias de ML en el dispositivo):
  - Solo se considera un modelo LINEAL (Probit / Regresion Logistica).
    K-NN queda descartado por diseno: requeriria guardar TODO el dataset de
    entrenamiento en la ESP32 (RAM ~320KB), mientras que un modelo lineal de
    5 clases x 7 features cabe en ~40 floats (~160 bytes), sin importar que
    tan grande sea el dataset de entrenamiento.
  - El grid search por lo tanto no busca "reducir tamano" (ya es minimo),
    sino la mejor regularizacion/generalizacion para que el modelo sea
    robusto al ruido del sensor en tiempo real.
"""

import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import statsmodels.api as sm

FEATURES = ["Accel_x", "Accel_y", "Accel_z", "w_x", "w_y", "w_z"]
# NOTA: Temp se excluye a proposito. En la sesion de captura la temperatura
# sube monotonicamente por autocalentamiento del sensor a medida que pasa el
# tiempo (Mov00->Mov04 fueron capturados en ese orden), asi que Temp termina
# correlacionada con el ORDEN de captura, no con el movimiento en si. Un
# modelo que la use como feature falla en vivo apenas la placa lleva mas
# tiempo encendida y la temperatura sale del rango visto en entrenamiento.
# Temp se sigue reportando en data.csv (Feature7) pero no entra al clasificador.
CLASSES = ["Mov00", "Mov01", "Mov02", "Mov03", "Mov04"]

RANDOM_STATE = 42


def load_data(path="training_data.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES].values
    y = df["Category"].values
    return X, y


def train_probit_ovr(X_train, y_train_idx, X_test):
    """Probit one-vs-rest: un modelo binario por clase, luego normalizado
    entre las 5 clases para que las probabilidades sumen 1 (igual formato
    que las columnas Prob MovXX pedidas en la guia)."""
    Xc_train = sm.add_constant(X_train, has_constant="add")
    Xc_test = sm.add_constant(X_test, has_constant="add")

    params = np.zeros((len(CLASSES), Xc_train.shape[1]))
    for c in range(len(CLASSES)):
        y_bin = (y_train_idx == c).astype(int)
        model = sm.Probit(y_bin, Xc_train)
        try:
            res = model.fit(disp=0, maxiter=200)
        except Exception:
            res = model.fit(disp=0, method="bfgs", maxiter=200)
        params[c, :] = res.params

    raw_probs = np.column_stack(
        [norm.cdf(Xc_test @ params[c, :]) for c in range(len(CLASSES))]
    )
    probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    return probs, params


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*[Pp]erfect separation.*")
    X, y = load_data()
    y_idx = np.array([CLASSES.index(v) for v in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, stratify=y_idx, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------------------------------------------------------------
    # 1) Probit one-vs-rest
    # ---------------------------------------------------------------
    print("=" * 70)
    print("PROBIT (one-vs-rest)")
    print("=" * 70)
    probit_probs, probit_params = train_probit_ovr(X_train_s, y_train, X_test_s)
    probit_pred = probit_probs.argmax(axis=1)
    probit_acc = accuracy_score(y_test, probit_pred)
    probit_f1 = f1_score(y_test, probit_pred, average="macro")
    print(f"Accuracy: {probit_acc:.4f}  |  Macro-F1: {probit_f1:.4f}")
    print(classification_report(y_test, probit_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=real, cols=predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, probit_pred), index=CLASSES, columns=CLASSES))

    # ---------------------------------------------------------------
    # 2) Logistic Regression + grid search
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION (grid search)")
    print("=" * 70)

    param_grid = [
        {
            "solver": ["lbfgs"],
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10, 100],
            "class_weight": [None, "balanced"],
        },
        {
            "solver": ["saga"],
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 10, 100],
            "class_weight": [None, "balanced"],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)

    print(f"Best CV macro-F1: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    best_lr = grid.best_estimator_
    lr_pred = best_lr.predict(X_test_s)
    lr_probs = best_lr.predict_proba(X_test_s)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average="macro")
    print(f"\nHeld-out test accuracy: {lr_acc:.4f}  |  Macro-F1: {lr_f1:.4f}")
    print(classification_report(y_test, lr_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=real, cols=predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, lr_pred), index=CLASSES, columns=CLASSES))

    # ---------------------------------------------------------------
    # 3) Elegir el mejor modelo (regla de la guia: Probit primero, LR si
    #    Probit no da buenos resultados) y exportar sus parametros.
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SELECCION DE MODELO")
    print("=" * 70)
    print(f"Probit   -> acc={probit_acc:.4f}  f1_macro={probit_f1:.4f}")
    print(f"LogReg   -> acc={lr_acc:.4f}  f1_macro={lr_f1:.4f}")

    # Con datos tan separables, ambos modelos pueden empatar en f1/accuracy,
    # pero Probit sin regularizar sufre "perfect separation": sus
    # coeficientes divergen (ver PerfectSeparationWarning) y sus
    # probabilidades se saturan a ~0/~1 en vez de dar una probabilidad real.
    # Por eso solo elegimos Probit si es CLARAMENTE mejor; en empate o si
    # sus coeficientes son inestables, preferimos la Regresion Logistica
    # regularizada (mas robusta para hardware con ruido de sensor real).
    probit_coef_norm = float(np.abs(probit_params[:, 1:]).max())
    margin = 0.01
    probit_unstable = probit_coef_norm > 5.0
    use_probit = (probit_f1 > lr_f1 + margin) and not probit_unstable
    chosen = "probit" if use_probit else "logistic_regression"
    print(f"Probit max |coef| = {probit_coef_norm:.2f} (>5 => senal de perfect separation)")
    print(f"\n>> Modelo elegido: {chosen}")

    if use_probit:
        # params: (5, 8) -> columna 0 es el intercepto (por add_constant)
        intercepts = probit_params[:, 0].tolist()
        coefficients = probit_params[:, 1:].tolist()
        link = "probit"
    else:
        intercepts = best_lr.intercept_.tolist()
        coefficients = best_lr.coef_.tolist()
        link = "softmax" if best_lr.get_params()["solver"] != "liblinear" and len(CLASSES) > 2 else "sigmoid_ovr"

    export = {
        "model_type": chosen,
        "link_function": link,
        "classes": CLASSES,
        "features": FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": coefficients,  # shape (5, 7), en el orden de FEATURES (ya escaladas)
        "intercepts": intercepts,      # shape (5,)
        "test_accuracy": probit_acc if use_probit else lr_acc,
        "test_macro_f1": probit_f1 if use_probit else lr_f1,
        "best_lr_grid_params": grid.best_params_,
    }

    with open("model_params.json", "w") as f:
        json.dump(export, f, indent=2)
    print("\nParametros exportados a model_params.json")


if __name__ == "__main__":
    main()
