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

1. ~~**Repo & environment setup**~~ — GitHub repo, PlatformIO scaffold, reused assets copied in.
2. ~~**Hardware bring-up**~~ — MPU6050 + OLED verified (found and fixed a real wiring issue on the OLED).
3. **Data capture** — *skipped*: reused `data/training_data.csv` from `movedge` as-is.
4. ~~**Preprocessing**~~ — `StandardScaler` fit in Python, applied before the model boundary (see REPORT.md for why it can't live inside the TF graph without breaking int8 quantization).
5. ~~**Model training in TensorFlow**~~ — small Keras MLP, 100% held-out test accuracy.
6. ~~**Conversion to TensorFlow Lite**~~ — float32 and int8 variants (`training/convert_to_tflite.py`).
7. ~~**Optimization comparison**~~ — found & fixed an int8 quantization bug from mixed-scale features; documented size/accuracy tradeoffs in `models/tflite_comparison.json` and REPORT.md.
8. ~~**On-device inference (TFLite Micro)**~~ — int8 model embedded and running live on the ESP32 (`firmware/src/main.cpp`).
9. ~~**Test protocol execution**~~ — full guide routine (9×15s structured + 6×5s random) run interactively via `protocol/run_test_protocol.py`. **85.85% accuracy / 0.8517 macro-F1** — consistent with `movedge`'s own delivered result (87.18%).
10. ~~**Results & report**~~ — see `REPORT.md` and `deliverables/` (`main.cpp` + `data.csv` in the guide's exact format).
11. **Commit checkpoints** — done throughout; see commit history for the stage-by-stage progression.

See `REPORT.md` for the full writeup (preprocessing rationale, model definition, quantization comparison, protocol results, confusion matrix, and lessons learned).
