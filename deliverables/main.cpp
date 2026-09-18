// Step 8: real-time movement classification on the ESP32 using
// TensorFlow Lite for Microcontrollers. Replaces the previous bring-up
// sketch (see firmware/reference/main_reference_movedge.cpp for the
// movedge version, which used a hand-rolled softmax instead of TFLite).
//
// Model: models/movement_classifier_int8.tflite (int8 weights/activations,
// float32 I/O -- quantize/dequantize happen inside the graph, see
// training/convert_to_tflite.py), embedded as firmware/include/model_data.h.
// Scaling: firmware/include/movement_scaler.h replicates the same
// StandardScaler fit in training/train_tf_model.py.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "movement_scaler.h"

// ---------------- Hardware config (matches movedge / the guide) ----------------
#define I2C_SDA 21
#define I2C_SCL 22
#define OLED_WIDTH 128
#define OLED_HEIGHT 32
#define OLED_ADDR 0x3C

// Debe coincidir con la tasa usada al capturar data/training_data.csv.
#define SAMPLE_PERIOD_MS 10
#define N_RAW_FEATURES 7 // Accel_x,y,z, w_x,y,z, Temp (Temp se loguea pero no entra al modelo)

Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
bool oled_present = false;

// ---------------- TensorFlow Lite Micro globals ----------------
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* tfl_input = nullptr;
TfLiteTensor* tfl_output = nullptr;

// Determinado por prueba (patron usual en TFLite Micro): el modelo es
// diminuto (~4.5KB, MLP de 2 capas de 16 unidades), pero AllOpsResolver
// mas los buffers de quantize/dequantize necesitan margen.
constexpr int kTensorArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ---------------- Tipos compartidos entre tareas ----------------
struct SensorSample {
  float raw[N_RAW_FEATURES];
};

struct ClassificationResult {
  float raw[N_RAW_FEATURES];
  float probs[MC_N_CLASSES];
  int predicted_class;
  uint32_t seq;
};

static QueueHandle_t xSampleQueue;
static SemaphoreHandle_t xResultMutex;
static ClassificationResult sharedResult = {};

// ---------------- Tarea 1: Captura de Datos ----------------
void vTaskAdquisicionDatos(void *pvParameters) {
  for (;;) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    SensorSample sample;
    sample.raw[0] = a.acceleration.x;
    sample.raw[1] = a.acceleration.y;
    sample.raw[2] = a.acceleration.z;
    sample.raw[3] = g.gyro.x;
    sample.raw[4] = g.gyro.y;
    sample.raw[5] = g.gyro.z;
    sample.raw[6] = temp.temperature;

    if (xQueueSend(xSampleQueue, &sample, 0) != pdTRUE) {
      SensorSample discard;
      xQueueReceive(xSampleQueue, &discard, 0);
      xQueueSend(xSampleQueue, &sample, 0);
    }

    vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
  }
}

// ---------------- Tarea 2: Inferencia (TFLite Micro) ----------------
void vTaskInferencia(void *pvParameters) {
  SensorSample sample;
  for (;;) {
    if (xQueueReceive(xSampleQueue, &sample, portMAX_DELAY) == pdTRUE) {
      float scaled[MC_N_FEATURES];
      // Solo Accel_x/y/z, w_x/y/z entran al modelo (sin Temp, ver arriba).
      mc_scale_features(sample.raw, scaled);

      for (int i = 0; i < MC_N_FEATURES; i++) {
        tfl_input->data.f[i] = scaled[i];
      }

      uint32_t t0 = micros();
      TfLiteStatus invoke_status = interpreter->Invoke();
      uint32_t inference_us = micros() - t0;

      if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke() failed");
        continue;
      }

      float probs[MC_N_CLASSES];
      int predicted = 0;
      for (int c = 0; c < MC_N_CLASSES; c++) {
        probs[c] = tfl_output->data.f[c];
        if (probs[c] > probs[predicted]) predicted = c;
      }

      if (xSemaphoreTake(xResultMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        memcpy(sharedResult.raw, sample.raw, sizeof(sample.raw));
        memcpy(sharedResult.probs, probs, sizeof(probs));
        sharedResult.predicted_class = predicted;
        sharedResult.seq++;
        xSemaphoreGive(xResultMutex);
      }

      static uint32_t log_counter = 0;
      if (++log_counter >= 200) { // ~cada 2s con SAMPLE_PERIOD_MS=10
        log_counter = 0;
        Serial.print("[inference] ");
        Serial.print(inference_us);
        Serial.println(" us");
      }
    }
  }
}

// ---------------- Tarea 3: Visualizacion ----------------
void vTaskVisualizacion(void *pvParameters) {
  uint32_t last_seq_seen = 0;
  uint32_t oled_counter = 0;
  const uint32_t OLED_REFRESH_EVERY = 20; // ~cada 200 ms con SAMPLE_PERIOD_MS=10

  // Encabezado del log por serial, formato esperado por protocol/run_test_protocol.py.
  Serial.println("Accel_x;Accel_y;Accel_z;w_x;w_y;w_z;Temp;ClasePredecida;"
                  "ProbMov00;ProbMov01;ProbMov02;ProbMov03;ProbMov04");

  for (;;) {
    ClassificationResult local;
    bool has_new = false;

    if (xSemaphoreTake(xResultMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
      if (sharedResult.seq != last_seq_seen) {
        local = sharedResult;
        last_seq_seen = sharedResult.seq;
        has_new = true;
      }
      xSemaphoreGive(xResultMutex);
    }

    if (has_new) {
      for (int i = 0; i < N_RAW_FEATURES; i++) {
        Serial.print(local.raw[i], 4);
        Serial.print(";");
      }
      Serial.print(MC_CLASS_NAMES[local.predicted_class]);
      for (int c = 0; c < MC_N_CLASSES; c++) {
        Serial.print(";");
        Serial.print(local.probs[c], 6);
      }
      Serial.println();

      oled_counter++;
      if (oled_present && oled_counter >= OLED_REFRESH_EVERY) {
        oled_counter = 0;
        display.clearDisplay();
        display.setCursor(0, 0);
        display.setTextSize(2);
        display.println(MC_CLASS_NAMES[local.predicted_class]);
        display.setTextSize(1);
        display.print("p=");
        display.println(local.probs[local.predicted_class], 2);
        display.display();
      }
    }

    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Wire.begin(I2C_SDA, I2C_SCL);

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  Serial.println("MPU6050 OK");

  oled_present = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oled_present) {
    Serial.println("SSD1306 no encontrado (revise conexion I2C); continuando solo con Serial.");
  } else {
    Serial.println("SSD1306 OK");
    display.clearDisplay();
    display.display();
  }

  // ---------------- TFLite Micro init ----------------
  tfl_model = tflite::GetModel(g_movement_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1) { delay(10); }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) { delay(10); }
  }

  tfl_input = interpreter->input(0);
  tfl_output = interpreter->output(0);
  Serial.print("TFLite Micro ready. Arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(" / ");
  Serial.println(kTensorArenaSize);

  xSampleQueue = xQueueCreate(10, sizeof(SensorSample));
  xResultMutex = xSemaphoreCreateMutex();

  xTaskCreate(vTaskAdquisicionDatos, "Captura", 4096, NULL, 3, NULL);
  xTaskCreate(vTaskInferencia, "Inferencia", 8192, NULL, 2, NULL);
  xTaskCreate(vTaskVisualizacion, "Visualizacion", 4096, NULL, 1, NULL);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
