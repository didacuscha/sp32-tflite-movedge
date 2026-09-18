// Step 2/3: hardware bring-up + data acquisition source.
// No classifier yet -- this just confirms the MPU6050/OLED wiring and
// streams raw sensor readings over serial for training/data_acquisition.py
// to capture. The TFLite Micro inference pipeline replaces/extends this
// in step 8.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---------------- Hardware config (matches movedge / the guide) ----------------
#define I2C_SDA 21
#define I2C_SCL 22
#define OLED_WIDTH 128
#define OLED_HEIGHT 32
#define OLED_ADDR 0x3C

// Must match the rate assumed when interpreting captured data.csv timestamps.
#define SAMPLE_PERIOD_MS 10
#define OLED_REFRESH_EVERY 15 // ~every 150 ms, so the I2C bus isn't saturated

Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
bool oled_present = false;

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
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("MPU6050 OK");
    display.println("Streaming...");
    display.display();
    delay(800);
  }

  // Header for training/data_acquisition.py to skip (not 7 parseable floats).
  Serial.println("Accel_x;Accel_y;Accel_z;w_x;w_y;w_z;Temp");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.print(a.acceleration.x, 4); Serial.print(";");
  Serial.print(a.acceleration.y, 4); Serial.print(";");
  Serial.print(a.acceleration.z, 4); Serial.print(";");
  Serial.print(g.gyro.x, 4); Serial.print(";");
  Serial.print(g.gyro.y, 4); Serial.print(";");
  Serial.print(g.gyro.z, 4); Serial.print(";");
  Serial.println(temp.temperature, 4);

  static uint32_t oled_counter = 0;
  if (oled_present && (++oled_counter >= OLED_REFRESH_EVERY)) {
    oled_counter = 0;
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.print("Ax "); display.println(a.acceleration.x, 2);
    display.print("Ay "); display.println(a.acceleration.y, 2);
    display.print("Az "); display.println(a.acceleration.z, 2);
    display.display();
  }

  delay(SAMPLE_PERIOD_MS);
}
