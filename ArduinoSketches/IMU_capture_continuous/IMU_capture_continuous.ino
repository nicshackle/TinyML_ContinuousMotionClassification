/*

  A sketch to capture continuous motion data,
  such that machine learning models can be trained on it.

  Start the recording by sending an 's' through the serial console.
  Copy the output of the serial console to a CSV file, alongside the python training script.

  Target board: Arduino Nano 33 BLE Sense board

*/

#include <Arduino_LSM9DS1.h>

// How many data points are required to capture a few periods of the motion
#define SAMPLE_SIZE 119

// How many samples of the same motion would you like to capture?
// The higher the number, the more accurate the model will be
#define SAMPLES_REQUIRED 10

int recordsRequired =  SAMPLE_SIZE * SAMPLES_REQUIRED;

void setup() {
  Serial.begin(11500);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("started");
  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, 1);

}

// start at -20 to allow rig to settle (e.g. after being reset)
int recordsTaken = -20;
bool triggered = false;

void loop() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == 's') {
      triggered = true;
      recordsTaken = -20;
    }
  }

  if (triggered) {
    digitalWrite(LEDR, 0);
    // print the CSV header
    Serial.println("aX,aY,aZ,gX,gY,gZ");

    while (recordsTaken < recordsRequired) {
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        float aX, aY, aZ, gX, gY, gZ;
        IMU.readAcceleration(aX, aY, aZ);
        IMU.readGyroscope(gX, gY, gZ);

        if (recordsTaken >= 0) {
          Serial.print(aX, 3);
          Serial.print(',');
          Serial.print(aY, 3);
          Serial.print(',');
          Serial.print(aZ, 3);
          Serial.print(',');
          Serial.print(gX, 3);
          Serial.print(',');
          Serial.print(gY, 3);
          Serial.print(',');
          Serial.print(gZ, 3);
          Serial.println();
        }
        recordsTaken++;
      } // close if
    } //close while

    digitalWrite(LEDR, 1);
    triggered = false;
  }
}
