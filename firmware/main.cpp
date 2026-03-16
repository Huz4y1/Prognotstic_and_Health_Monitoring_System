#include <Wire.h>
#include <MPU6050.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>


//Motor TB6612FNG
#define PWMA 5
#define AIN1 18
#define AIN2 19
#define STBY 23

//Pushbuttons
#define BTN_UP 12
#define BTN_DOWN 13

//Sensors
#define TEMP_PIN 4
#define CURRENT_PIN 34
#define HALL_PIN 27

//Optional GPS
#define GPS_RX 17
#define GPS_TX 16



MPU6050 imu;
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

volatile int pulseCount = 0;

float vibrationBuffer[50];
int vibIndex = 0;

float lastTemp = 0;
float tempSlope = 0;

float lastCurrent = 0;
float currentTrend = 0;

float lastRoll = 0;
float lastPitch = 0;
float lastYaw = 0;

float orientationJitter = 0;

unsigned long lastTime = 0;

//Motor PWM
int speed = 100; // 0-255



void IRAM_ATTR hallISR(){
  pulseCount++;
}


float computeRMS(){
  float sum = 0;
  for(int i=0;i<50;i++)
    sum += vibrationBuffer[i]*vibrationBuffer[i];
  return sqrt(sum/50);
}

float readCurrent(){
  int raw = analogRead(CURRENT_PIN);
  float voltage = raw * (3.3/4095.0);
  float current = (voltage - 2.5)/0.185; 
  return current;
}

float getRPM(){
  int pulses = pulseCount;
  pulseCount = 0;
  return pulses * 60; 
}



void setup(){
  Serial.begin(115200);
  Wire.begin();
  imu.initialize();
  tempSensor.begin();

  pinMode(HALL_PIN, INPUT);
  attachInterrupt(HALL_PIN, hallISR, RISING);

  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH); 

  pinMode(BTN_UP, INPUT_PULLDOWN);
  pinMode(BTN_DOWN, INPUT_PULLDOWN);

  
  digitalWrite(AIN1,HIGH);
  digitalWrite(AIN2,LOW);
  analogWrite(PWMA, speed);

  lastTime = millis();
}



void loop(){
  unsigned long now = millis();
  float dt = (now - lastTime)/1000.0;


  if(digitalRead(BTN_UP)==HIGH && speed < 255){
    speed += 5;
    if(speed>255) speed=255;
    analogWrite(PWMA, speed);
    delay(100);
  }
  if(digitalRead(BTN_DOWN)==HIGH && speed > 0){
    speed -= 5;
    if(speed<0) speed=0;
    analogWrite(PWMA, speed);
    delay(100);
  }

  int16_t ax,ay,az,gx,gy,gz;
  imu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);
  float accelMag = sqrt(ax*ax + ay*ay + az*az);
  vibrationBuffer[vibIndex] = accelMag;
  vibIndex = (vibIndex+1)%50;

  float vibrationRMS = computeRMS();

  float roll = gx/131.0;
  float pitch = gy/131.0;
  float yaw = gz/131.0;

  orientationJitter = abs(roll-lastRoll)+abs(pitch-lastPitch)+abs(yaw-lastYaw);
  lastRoll=roll; lastPitch=pitch; lastYaw=yaw;


  tempSensor.requestTemperatures();
  float temp = tempSensor.getTempCByIndex(0);
  tempSlope = (temp - lastTemp)/dt;
  lastTemp = temp;

 
  float current = readCurrent();
  currentTrend = current - lastCurrent;
  lastCurrent = current;

  
  float rpm = getRPM();

  lastTime = now;


  StaticJsonDocument<512> doc;
  doc["timestamp"] = now;
  doc["temp"] = temp;
  doc["temp_slope"] = tempSlope;
  doc["vibration_rms"] = vibrationRMS;
  doc["current"] = current;
  doc["current_trend"] = currentTrend;
  doc["rpm"] = rpm;
  doc["roll"] = roll;
  doc["pitch"] = pitch;
  doc["yaw"] = yaw;
  doc["orientation_jitter"] = orientationJitter;
  doc["motor_pwm"] = speed;



  serializeJson(doc, Serial);
  Serial.println();

  delay(50); 
}
