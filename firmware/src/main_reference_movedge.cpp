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

#include "movement_classifier.h"

// ---------------- Configuracion de hardware ----------------
#define I2C_SDA 21
#define I2C_SCL 22
#define OLED_WIDTH 128
#define OLED_HEIGHT 32
#define OLED_ADDR 0x3C

// Debe coincidir con la tasa usada al capturar data.csv (ver data_acquisition.py)
#define SAMPLE_PERIOD_MS 10

Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);

// ---------------- Tipos compartidos entre tareas ----------------
// Se guardan las 7 lecturas crudas (para el log/CSV, Feature1..Feature7 de
// la guia), pero solo las primeras MC_N_FEATURES (Accel x/y/z, gyro x/y/z)
// se usan como entrada del clasificador: Temp se excluye a proposito porque
// esta confundida con el orden/tiempo de captura (ver train_classifier.py).
#define N_RAW_FEATURES 7

struct SensorSample {
  float raw[N_RAW_FEATURES]; // Accel_x,Accel_y,Accel_z,w_x,w_y,w_z,Temp
};

struct ClassificationResult {
  float raw[N_RAW_FEATURES];
  float probs[MC_N_CLASSES];
  int predicted_class;
  uint32_t seq; // se incrementa en cada clasificacion nueva
};

// ---------------- FreeRTOS: 1 cola + 1 semaforo (requisito minimo) ----------------
static QueueHandle_t xSampleQueue;      // vTaskAdquisicionDatos -> vTaskCalculoProbabilidades
static SemaphoreHandle_t xResultMutex;  // protege sharedResult (Clasificacion <-> Visualizacion)
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

    // Si la cola esta llena, se descarta la muestra mas vieja: preferimos
    // datos frescos en tiempo real antes que acumular latencia.
    if (xQueueSend(xSampleQueue, &sample, 0) != pdTRUE) {
      SensorSample discard;
      xQueueReceive(xSampleQueue, &discard, 0);
      xQueueSend(xSampleQueue, &sample, 0);
    }

    vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
  }
}

// ---------------- Tarea 2: Normalizacion / Clasificacion ----------------
void vTaskCalculoProbabilidades(void *pvParameters) {
  SensorSample sample;
  for (;;) {
    if (xQueueReceive(xSampleQueue, &sample, portMAX_DELAY) == pdTRUE) {
      float probs[MC_N_CLASSES];
      // mc_classify() ya incluye la normalizacion (estandarizacion) y el
      // calculo de la Regresion Logistica (softmax) entrenada offline.
      // Solo se pasan las primeras MC_N_FEATURES (sin Temp, ver arriba).
      int predicted = mc_classify(sample.raw, probs);

      if (xSemaphoreTake(xResultMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        memcpy(sharedResult.raw, sample.raw, sizeof(sample.raw));
        memcpy(sharedResult.probs, probs, sizeof(probs));
        sharedResult.predicted_class = predicted;
        sharedResult.seq++;
        xSemaphoreGive(xResultMutex);
      }
    }
  }
}

// ---------------- Tarea 3: Visualizacion ----------------
void vTaskVisualizacion(void *pvParameters) {
  uint32_t last_seq_seen = 0;
  uint32_t oled_counter = 0;
  const uint32_t OLED_REFRESH_EVERY = 20; // ~cada 200 ms con SAMPLE_PERIOD_MS=10

  // Encabezado del log por serial. No incluye "Clase Real": eso lo agrega
  // el script de host (fase 6), que es quien sabe que movimiento se pidio.
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
      // --- Serial: una linea por muestra clasificada (log para fase 6) ---
      // Se imprimen las 7 lecturas crudas (incl. Temp = Feature7 de la
      // guia), aunque el clasificador solo haya usado las primeras 6.
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

      // --- OLED: refresco mas lento para no saturar el bus I2C ---
      oled_counter++;
      if (oled_counter >= OLED_REFRESH_EVERY) {
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

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 no encontrado (revise conexion I2C); continuando solo con Serial.");
  } else {
    display.clearDisplay();
    display.display();
  }

  xSampleQueue = xQueueCreate(10, sizeof(SensorSample));
  xResultMutex = xSemaphoreCreateMutex();

  xTaskCreate(vTaskAdquisicionDatos,       "Captura",       4096, NULL, 3, NULL);
  xTaskCreate(vTaskCalculoProbabilidades, "Clasificacion", 4096, NULL, 2, NULL);
  xTaskCreate(vTaskVisualizacion, "Visualizacion", 4096, NULL, 1, NULL);
}

void loop() {
  // Todo el trabajo ocurre en las tareas de FreeRTOS creadas en setup().
  vTaskDelay(pdMS_TO_TICKS(1000));
}
