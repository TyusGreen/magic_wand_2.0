// CPE 470 Final Project: Magic Wand 2.0
// Tyus Green, Danny Moreno, Noah De La Pena
// Most of this code is copied directly from the Harvard_TinyML "magic_wand" example sketch
// When the user makes a gesture with the arduino in the shape of a star,
// the piezo buzzer or on-board LED will turn on for 1 second.
// The buzzer needs to be connected from Digital Pin 4 to GND.

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "magic_wand_model_data.h"
#include "rasterize_stroke.h"
#include "imu_provider.h"

#define BLE_SENSE_UUID(val) ("4798e0f2-" val "-4d68-af64-8a8f5258404e")

namespace {

  const int VERSION = 0x00000000;

  // Constants for image rasterization
  constexpr int raster_width = 32;
  constexpr int raster_height = 32;
  constexpr int raster_channels = 3;
  constexpr int raster_byte_count = raster_height * raster_width * raster_channels;
  int8_t raster_buffer[raster_byte_count];

  // BLE settings
  BLEService        service                       (BLE_SENSE_UUID("0000"));
  BLECharacteristic strokeCharacteristic          (BLE_SENSE_UUID("300a"), BLERead, stroke_struct_byte_count);
  
  // String to calculate the local and device name
  String name;
  
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 30 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  
  // -------------------------------------------------------------------------------- //
  // UPDATE THESE VARIABLES TO MATCH THE NUMBER AND LIST OF GESTURES IN YOUR DATASET  //
  // -------------------------------------------------------------------------------- //
  constexpr int label_count = 10;
  const char* labels[label_count] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

}  // namespace

#define BUZZER_PIN 4  // Define the buzzer pin

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  Serial.println("Started");

  pinMode(BUZZER_PIN, OUTPUT);  // Initialize the buzzer pin
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialized IMU!");
    while (1);
  }
  SetupIMU();

  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1);
  }

  // BLE setup (unchanged)
  // TensorFlow Lite setup (unchanged)

  interpreter->AllocateTensors();
}

void loop() {
  BLEDevice central = BLE.central();
  
  static bool was_connected_last = false;
  if (central && !was_connected_last) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());
  }
  was_connected_last = central;

  const bool data_available = IMU.accelerationAvailable() || IMU.gyroscopeAvailable();
  if (!data_available) {
    return;
  }
  
  int accelerometer_samples_read;
  int gyroscope_samples_read;
  ReadAccelerometerAndGyroscope(&accelerometer_samples_read, &gyroscope_samples_read);

  bool done_just_triggered = false;
  if (gyroscope_samples_read > 0) {
    EstimateGyroscopeDrift(current_gyroscope_drift);
    UpdateOrientation(gyroscope_samples_read, current_gravity, current_gyroscope_drift);
    UpdateStroke(gyroscope_samples_read, &done_just_triggered);
  }
  
  if (accelerometer_samples_read > 0) {
    EstimateGravityDirection(current_gravity);
    UpdateVelocity(accelerometer_samples_read, current_gravity);
  }

  if (done_just_triggered) {
    RasterizeStroke(stroke_points, *stroke_transmit_length, 0.6f, 0.6f, raster_width, raster_height, raster_buffer);

    TfLiteTensor* model_input = interpreter->input(0);
    for (int i = 0; i < raster_byte_count; ++i) {
      model_input->data.int8[i] = raster_buffer[i];
    }
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }
    
    TfLiteTensor* output = interpreter->output(0);
    int8_t max_score;
    int max_index;
    for (int i = 0; i < label_count; ++i) {
      const int8_t score = output->data.int8[i];
      if ((i == 0) || (score > max_score)) {
        max_score = score;
        max_index = i;
      }
    }

    if (strcmp(labels[max_index], "star") == 0) {  // Check if detected label is "star"
      TF_LITE_REPORT_ERROR(error_reporter, "Detected a star!");
      // If Piexo Buzzer works, uncomment code below
      // playSound();

      // This is backup code because our piezo speaker stopped workingc:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\CPE470_FinalProject.ino c:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\imu_provider.h c:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\LICENSE c:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\magic_wand_model_data.cpp c:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\magic_wand_model_data.h c:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\rasterize_stroke.cpp c:\Users\tyusg\Documents\Arduino\CPE470_FinalProject\rasterize_stroke.h
      digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
      delay(1000);
      digitalWrite(LED_BUILTIN, LOW);  // turn the LED on (HIGH is the voltage level)
    }
  }
}

// Function to play buzzer sound.
void playSound() {
  tone(BUZZER_PIN, 1000, 1000);  // Play 1 kHz tone for 1000 ms
  delay(1000);                   // Wait for the tone to finish
}