
#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

const int bufferSize = 119;

int samplesRead = 0;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* MOTIONS[] = {
  "A",
  "B"
};

// The on-board RGB LED is active low, so these are 
// to make digitalWrite's more readable
#define ON 0
#define OFF 1

#define NUM_MOTIONS (sizeof(MOTIONS) / sizeof(MOTIONS[0]))

void setup() {
  Serial.begin(115200);
  //while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  pinMode(LEDR, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, OFF);
  digitalWrite(LEDB, OFF);

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}



void loop() {
  while (samplesRead < bufferSize) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
       float aX, aY, aZ, gX, gY, gZ;
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      /*
       * A note on loading IMU data into the tensors
       * 
       * Depending on the motion captured, you may not need all of the data
       * that the IMU provides. Omit axes/dimensions as needed below.
       * 
       * Data loaded into the tensors should be normalized to the range of 0-1.
       * Normalize such that your data fills the whole range of 0-1.
       * Make sure that the normalization happening below matches that of the
       * training script.
       * 
       * Example of full-range normalizations (good for drastic motions)
       * C++: tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
       * Python: (float(df['aX'][index]) + 4) / 8,
       * 
       * Example of smaller-range normalizations (good for subtle motions)
       * C++: tflInputTensor->data.f[samplesRead] = (aY - 0.5);
       * Python: (float(df['aY'][index]) - 0.5)
       * 
       * If just one dimension is being used, no arithmatic is required in the [...]
       * Example of single axis:
       * tflInputTensor->data.f[samplesRead] = (aY - 0.5);
       * 
       * Example of multi-dimension:
       * tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
       * tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
       * tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
       * tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
       * tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
       * tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;
       */

       // this example just uses one axis
       tflInputTensor->data.f[samplesRead] = (aY / 0.8);

      samplesRead++;

      if (samplesRead == bufferSize) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Print results
        for (int i = 0; i < NUM_MOTIONS; i++) {
          Serial.print(MOTIONS[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);
        }

        Serial.print("detected ");
        if(tflOutputTensor->data.f[0] > 0.8){
          Serial.println(MOTIONS[0]);
          digitalWrite(LEDG, ON);
          digitalWrite(LEDR, OFF);
        } else if(tflOutputTensor->data.f[1] > 0.8){
          Serial.println(MOTIONS[1]);
          digitalWrite(LEDG, OFF);
          digitalWrite(LEDR, ON);
        } 

        samplesRead = 0; // reset buffer
      }
    }
  }
}
