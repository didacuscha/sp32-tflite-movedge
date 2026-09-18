# sp32-tflite-movedge

Movement classifier on ESP32, trained with TensorFlow and deployed on-device with
**TensorFlow Lite for Microcontrollers**. This is the follow-up microproject to
[`movedge`](https://github.com/didacuscha/movedge), which solved the same
classification problem with a hand-rolled logistic-regression softmax hardcoded
into the firmware. This time the model is trained in TensorFlow, converted to
`.tflite` (float and quantized), and run on the ESP32 through the TFLite Micro
interpreter — with a comparison between optimization levels.

See `docs/guide.pdf` for the assignment brief.

## Hardware

- ESP32 DEVKIT V1
- MPU6050 (accelerometer + gyroscope)
- SSD1306 0.96" OLED (I2C, optional)
- Breadboard + jumpers

## Repo layout

```
docs/         assignment guides (this one + the previous project's, for reference)
data/         labeled sensor captures reused/extended from the previous project
training/     Python: data capture, TF model training, TFLite conversion/quantization
models/       committed .tflite artifacts (float + quantized) and the exported C header
firmware/     PlatformIO project (ESP32, Arduino framework, TFLite Micro interpreter)
protocol/     test protocol runner + results (controlled + randomized movement routine)
```

Files suffixed `_reference` (e.g. `train_classifier_reference.py`,
`main_reference_movedge.cpp`) are carried over unmodified from `movedge` for
comparison/reuse and are not part of the new pipeline.

## What's being reused from `movedge`

- Hardware wiring and pin config (I2C on 21/22, MPU6050 ranges/filter, OLED addr).
- `data_acquisition.py` capture protocol and CSV schema (`Accel_x/y/z, w_x/y/z, Temp`).
- Existing labeled datasets (`data/data.csv`, `data/training_data.csv`) as a
  starting point — likely to be extended with fresh captures.
- The FreeRTOS task split (acquisition / inference / display) as the firmware
  skeleton, swapping the hand-rolled `mc_classify()` for a TFLite Micro
  `Interpreter::Invoke()`.
- `run_test_protocol.py` and the verification-set tooling, adapted for the new
  output format.

## Implementation plan

1. **Repo & environment setup** *(done)* — GitHub repo, PlatformIO scaffold,
   reused assets copied in.
2. **Hardware bring-up** — wire ESP32 + MPU6050 (+ OLED), confirm serial output
   of the 7 raw features, matching the previous project's setup.
3. **Data capture** — reuse/extend `training/data_acquisition.py` to build a
   labeled dataset for Mov00–Mov04 (or custom gestures), following the
   controlled-routine protocol from the guide. Decide whether to reuse
   `data/data.csv` as-is or recapture for a larger/cleaner set.
4. **Preprocessing & feature engineering** — normalization/scaling, optional
   windowing (previous project used raw per-sample features; this is a good
   place to try more powerful features, e.g. rolling stats or magnitude,
   since we're no longer limited to a linear model computable by hand).
5. **Model training in TensorFlow** — build and train a small Keras model
   (dense network, or 1D-conv over a short window) on the labeled data;
   evaluate accuracy/F1/confusion matrix like the reference script did.
6. **Conversion to TensorFlow Lite** — convert the trained Keras model to
   `.tflite`, once as float32 and once with post-training quantization
   (dynamic range and/or full-int8 with a representative dataset).
7. **Optimization comparison** — benchmark size, latency, and accuracy of
   float vs. quantized models (on the host, e.g. with the TFLite Python
   interpreter) to produce the comparison the guide asks for.
8. **On-device inference (TFLite Micro)** — embed the chosen `.tflite` model
   as a C array in `firmware/`, wire it into a TFLite Micro
   `MicroInterpreter`, and replace `mc_classify()` in the FreeRTOS pipeline
   with real-time inference on the ESP32.
9. **Test protocol execution** — run the guide's controlled routine (5×15s
   movements) + randomized routine (30s in 5s windows), logging predicted
   class and inference score per sample to a `raw` file.
10. **Results & report** — assemble `data.csv` in the exact deliverable
    format (Feature1–7, Clase Real, Clase Predecida, Resultado Inferencia),
    and write the report covering preprocessing choices, model definition,
    and the optimization comparison metrics.
11. **Commit checkpoints** — commit after each stage above (data capture,
    training, conversion, firmware integration, protocol results) instead of
    one final dump, so progress is recoverable at every step.
