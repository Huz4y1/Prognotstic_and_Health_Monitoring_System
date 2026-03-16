#include <Arduino.h>

void    IRAM_ATTR hallISR();
float   computeVibrationRMS();
float   readCurrent();
float   getRPM(float dt);

#include <Wire.h>
#include <MPU6050.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

#define PWMA  5
#define AIN1  18
#define AIN2  19
#define STBY  23

#define BTN_UP   12
#define BTN_DOWN 13

#define TEMP_PIN    4
#define CURRENT_PIN 34
#define HALL_PIN    27

#define ACCEL_SCALE 16384.0f
#define GYRO_SCALE  131.0f

MPU6050 imu;
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

volatile int pulseCount = 0;

float vibrationBuffer[50];
int   vibIndex = 0;

int motorSpeed = 100;

unsigned long lastTime = 0;

void IRAM_ATTR hallISR() {
  pulseCount++;
}

float computeVibrationRMS() {
  float sum = 0;
  for (int i = 0; i < 50; i++)
    sum += vibrationBuffer[i] * vibrationBuffer[i];
  return sqrt(sum / 50.0f);
}

float readCurrent() {
  int   raw     = analogRead(CURRENT_PIN);
  float voltage = raw * (3.3f / 4095.0f);
  float current = (voltage - 2.5f) / 0.185f;
  return current;
}

float getRPM(float dt) {
  int pulses = pulseCount;
  pulseCount = 0;
  if (dt <= 0) return 0;
  return (pulses / dt) * 60.0f;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  imu.initialize();
  if (!imu.testConnection()) {
    Serial.println("{\"error\":\"MPU6050 connection failed\"}");
  }

  tempSensor.begin();

  pinMode(HALL_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN), hallISR, RISING);

  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, motorSpeed);

  pinMode(BTN_UP,   INPUT_PULLDOWN);
  pinMode(BTN_DOWN, INPUT_PULLDOWN);

  lastTime = millis();
}

void loop() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0f;
  lastTime = now;

  if (digitalRead(BTN_UP) == HIGH && motorSpeed < 255) {
    motorSpeed += 5;
    if (motorSpeed > 255) motorSpeed = 255;
    analogWrite(PWMA, motorSpeed);
    delay(150);
  }
  if (digitalRead(BTN_DOWN) == HIGH && motorSpeed > 0) {
    motorSpeed -= 5;
    if (motorSpeed < 0) motorSpeed = 0;
    analogWrite(PWMA, motorSpeed);
    delay(150);
  }

  int16_t ax, ay, az, gx, gy, gz;
  imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  float ax_g = ax / ACCEL_SCALE;
  float ay_g = ay / ACCEL_SCALE;
  float az_g = az / ACCEL_SCALE;
  float accelMag = sqrt(ax_g*ax_g + ay_g*ay_g + az_g*az_g) - 1.0f;
  if (accelMag < 0) accelMag = 0;

  vibrationBuffer[vibIndex] = accelMag;
  vibIndex = (vibIndex + 1) % 50;
  float vibration = computeVibrationRMS();

  float roll  = gx / GYRO_SCALE;
  float pitch = gy / GYRO_SCALE;
  float yaw   = gz / GYRO_SCALE;

  tempSensor.requestTemperatures();
  float temp = tempSensor.getTempCByIndex(0);
  if (temp == DEVICE_DISCONNECTED_C) temp = 0.0f;

  float current = readCurrent();

  float rpm = getRPM(dt);

  StaticJsonDocument<256> doc;
  doc["temp"]      = temp;
  doc["vibration"] = vibration;
  doc["current"]   = current;
  doc["rpm"]       = rpm;
  doc["roll"]      = roll;
  doc["pitch"]     = pitch;
  doc["yaw"]       = yaw;

  serializeJson(doc, Serial);
  Serial.println();

  delay(100);
}