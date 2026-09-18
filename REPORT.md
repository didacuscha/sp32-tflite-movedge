# TensorFlowLite-MovEdge — Informe

Clasificador de movimientos en ESP32, entrenado con TensorFlow y desplegado
con TensorFlow Lite for Microcontrollers. Repo: `sp32-tflite-movedge`.
Continuación de `movedge` (microproyecto anterior, clasificador con
regresión logística implementada a mano, sin TensorFlow).

## 1. Preprocesamiento

**Dataset:** reutilizado de `movedge` (`data/training_data.csv`, 9932
muestras etiquetadas: Mov00–Mov04). Se decidió no recapturar inicialmente
para avanzar más rápido; un problema de sincronización durante las pruebas
en vivo (ver sección 4) hizo temporalmente sospechar de un desajuste de
orientación del sensor, pero quedó descartado al repetir el protocolo de
forma interactiva (ver sección 4) — el dataset original seguía siendo
válido.

**Selección de features:** Accel_x, Accel_y, Accel_z, w_x, w_y, w_z. Se
excluye `Temp` (Feature7): en la sesión de captura original la temperatura
sube monótonamente por autocalentamiento del sensor a medida que pasa el
tiempo (las clases se capturaron en orden Mov00→Mov04), quedando
correlacionada con el *orden de captura* y no con el movimiento en sí. Un
modelo que la use falla en vivo apenas la placa lleva más tiempo encendida.

**Hallazgo importante:** estos "movimientos" son **poses estáticas**
(inclinación sostenida que redirige el vector de gravedad entre los ejes
del acelerómetro), no gestos dinámicos — el giroscopio se mantiene ~0 en
las 5 clases (ver medias en `data/training_data.csv`). Esto se descubrió
durante las pruebas en vivo, cuando movimientos rápidos tipo "swipe" no
disparaban ninguna clase distinta de Mov00: solo una inclinación sostenida
de ~90° reproduce el patrón de entrenamiento.

**Escalado:** `StandardScaler` (media/desviación por feature), ajustado en
Python sobre el set de entrenamiento (`training/train_tf_model.py`) y
aplicado ANTES de que las features entren al modelo — no como una capa
`Normalization` dentro del grafo de TensorFlow. Se probó la versión con
Normalization-en-el-grafo primero, y rompía la cuantización int8: Accel_*
tiene rango ~[-10,10] m/s² mientras que w_* tiene rango ~[-0.1,0.1] rad/s
(~100x más chico), y TFLite cuantiza int8 con una sola escala por tensor de
entrada — con features en rangos tan distintos esa escala queda dominada
por Accel y los ejes de giroscopio colapsan a 2-3 niveles distinguibles,
tumbando la accuracy a ~55%. Escalar en Python antes de la cuantización
resuelve el problema completamente (ver sección 3). Los parámetros del
scaler quedan en `models/scaler.json` y se replican en el firmware
(`firmware/include/movement_scaler.h`, autogenerado).

## 2. Modelo (TensorFlow)

MLP pequeño (`training/train_tf_model.py`): `Input(6) → Dense(16, relu) →
Dense(16, relu) → Dense(5, softmax)`. Entrenado con Adam +
sparse_categorical_crossentropy, early stopping sobre validación.

**Métricas en held-out test set** (20% del dataset, split estratificado):
accuracy = 1.0000, macro-F1 = 1.0000 — el dataset es muy separable
(las 5 poses son geométricamente muy distintas en el espacio de features),
por lo que este resultado no es sorprendente ni indica overfitting
detectable con este método de evaluación.

## 3. TensorFlow Lite: conversión y comparación de optimización

`training/convert_to_tflite.py` genera dos variantes desde el mismo modelo
Keras:

| Variante | Tamaño | Accuracy (test, host) | Notas |
|---|---|---|---|
| float32 | 4204 bytes | 1.0000 | Sin cuantizar. |
| int8 (full-integer) | 4456 bytes | 1.0000 | Pesos y activaciones int8; I/O float32 (quantize/dequantize dentro del grafo). |

**Comparación:** para un modelo de este tamaño, la cuantización int8 **no
reduce el tamaño del archivo** (+6%) — el overhead de metadata de
cuantización supera el ahorro de los pesos int8 cuando el modelo tiene tan
pocos parámetros. Esto contradice la expectativa típica de "cuantizar =
más chico", y es un resultado real, no un error: la ganancia de
cuantización solo se nota en modelos con muchos más parámetros. La
comparación de latencia relevante (ESP32, con FPU) se hizo en el firmware,
no en el host (ver sección 4).

**Bug encontrado y corregido:** la primera versión (con `Normalization`
dentro del grafo) hacía que el modelo int8 cayera a 55% de accuracy /
0.14 macro-F1 en el test set, por el problema de escala descrito en la
sección 1. Se diagnosticó comparando los rangos min/max reales de
Accel_* vs w_* en el dataset, y se corrigió escalando en Python antes de
la cuantización. Después de la corrección, int8 iguala a float32 (1.0/1.0).
**Se eligió la variante int8** para el firmware, por su relevancia
pedagógica directa con el enfoque de la guía (cuantización) y porque
TFLite Micro está optimizado para aritmética entera.

## 4. Implementación en la ESP32 (TFLite Micro)

Librería: `spaziochirale/Chirale_TensorFLowLite` (TFLite Micro para
Arduino/ESP32). El modelo int8 se embebe como arreglo de bytes en
`firmware/include/model_data.h` (autogenerado por
`training/export_firmware_headers.py`).

**Arquitectura FreeRTOS** (heredada de `movedge`, 3 tareas + 1 cola + 1
semáforo, mínimo pedido por la guía):
1. `vTaskAdquisicionDatos` — lee MPU6050 cada 10ms, encola la muestra.
2. `vTaskInferencia` — desencola, escala features (`mc_scale_features`),
   corre `interpreter->Invoke()` (TFLite Micro), publica el resultado bajo
   mutex.
3. `vTaskVisualizacion` — imprime cada muestra clasificada por serial
   (formato consumido por `protocol/run_test_protocol.py`) y refresca el
   OLED (opcional) con la clase predicha.

Memoria: arena de TFLite Micro de 20KB reservados, de los cuales solo
~1.1KB se usan realmente (`arena_used_bytes()`) — sobredimensionada
deliberadamente por seguridad, no ajustada a su mínimo. RAM total del
firmware: 14.8% de 320KB. Flash: 45.3% de 1.3MB (el runtime de TFLite
Micro es la mayor parte de ese uso, no el modelo en sí, que pesa ~4.5KB).

## 5. Protocolo de pruebas

Rutina exacta de la guía: estructurada (Mov00→Mov01→Mov00→Mov02→Mov00→
Mov03→Mov00→Mov04→Mov00, 15s cada bloque) + aleatoria (6 bloques de 5s,
30s totales). Ejecutada con `protocol/run_test_protocol.py` en modo
interactivo (con `input()` entre bloques, para acomodar la pose antes de
que arranque el cronómetro de cada bloque — crítico para la calidad del
resultado, ver nota más abajo).

**Resultado (15116 muestras, `deliverables/data.csv`):**

- **Accuracy global: 0.8585** | **Macro-F1: 0.8517**

```
Clase Predecida  Mov00  Mov01  Mov02  Mov03  Mov04
Clase Real
Mov00             6660    285    166    162    361
Mov01              167   1540    167      0      0
Mov02              333      0   1529      0      0
Mov03              166      0      0   1156      0
Mov04              332      0      0      0   2092
```

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Mov00 | 0.8697 | 0.8724 | 0.8710 |
| Mov01 | 0.8438 | 0.8218 | 0.8327 |
| Mov02 | 0.8212 | 0.8212 | 0.8212 |
| Mov03 | 0.8771 | 0.8744 | 0.8758 |
| Mov04 | 0.8528 | 0.8630 | 0.8579 |

Este resultado es consistente con el precedente del proyecto anterior
(`movedge`, mismo protocolo, mismo hardware): su `data.csv` entregado tenía
87.18% de accuracy real. Un resultado de ~85-87% en pruebas en vivo, con
confusiones concentradas en las transiciones hacia/desde Mov00, es el
comportamiento esperado de este protocolo — la accuracy de 1.0 del modelo
en el set de test (sección 2) refleja generalización dentro del *mismo*
dataset de captura, no robustez ante el ruido de sostener una pose a mano
en tiempo real.

**Nota metodológica importante:** un primer intento de este protocolo,
corrido en modo automático (tiempos anunciados por chat en vez de
`input()` real) dio solo 33% de accuracy y llevó a sospechar inicialmente
de un desajuste físico del sensor (posible remontaje durante el cableado
del OLED). Al repetir el protocolo en modo interactivo real —
posicionarse en la pose ANTES de iniciar el cronómetro de cada bloque, tal
como lo hacía el script original de `movedge`— la accuracy subió a 85.85%,
descartando el desajuste de sensor como causa: el problema real era la
falta de sincronización, no los datos ni el modelo. Lección: el
`input()` entre bloques no es una formalidad, es lo que separa una prueba
válida de una inválida en este protocolo.

**Calibración de confianza:** `Resultado Inferencia` (probabilidad softmax
de la clase predicha) es ~0.996 tanto para predicciones correctas
(media 0.9959) como incorrectas (media 0.9907) — el modelo cuantizado es
*overconfident*: su score de salida no es un buen indicador de si la
predicción es correcta. Esto es consistente con cuantización int8 agresiva
sobre un modelo pequeño con fronteras de decisión muy nítidas.

## 6. Entregables

- `deliverables/main.cpp` — firmware de la ESP32 (copia de
  `firmware/src/main.cpp`).
- `deliverables/data.csv` — resultado real del protocolo de pruebas, con
  el formato exacto pedido por la guía (Feature1-7, Clase Real, Clase
  Predecida, Resultado Inferencia).
- Este informe (`REPORT.md`).

## 7. Aprendizajes clave

1. Cuando las features tienen escalas físicas muy distintas (m/s² vs
   rad/s), escalar DENTRO del grafo de TF antes de cuantizar puede romper
   silenciosamente la cuantización int8 (calibración dominada por la
   feature de mayor rango). Escalar en Python, antes del límite de
   cuantización, lo evita.
2. Cuantización int8 no siempre reduce el tamaño del modelo — en modelos
   muy pequeños el overhead de metadata puede superar el ahorro de pesos.
3. Un protocolo de prueba con poses sostenidas necesita tiempo real de
   acomodo antes de que empiece a contar cada bloque; automatizar el
   tiempo sin esa confirmación degrada la accuracy medida sin que el
   modelo haya cambiado.
4. La confianza de salida de un modelo cuantizado no es automáticamente
   confiable como métrica de incertidumbre.
