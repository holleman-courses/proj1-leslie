#include "image_provider.h"
#include "flower_detect_model_data.h"
#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <TinyMLShield.h>

// Function to get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter *error_reporter, int image_width,
                      int image_height, int channels, int8_t *image_data)
{
  // Buffer to hold a QCIF grayscale image (176x144) from the camera
  byte data[176 * 144];

  static bool g_is_camera_initialized = false;

  // Initialize camera if not already initialized
  if (!g_is_camera_initialized)
  {
    // Attempt to initialize the camera
    if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675))
    {
      // If initialization fails, report the error
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true; // Mark the camera as initialized
  }

  // Capture a frame from the camera
  Camera.readFrame(data);

  // Define cropping area for a 96x96 image (centered in the 176x144 frame)
  int min_x = (176 - 96) / 2;
  int min_y = (144 - 96) / 2;
  int index = 0;

  // Crop and downscale the image to 96x96 (by selecting the center portion)
  for (int y = min_y; y < min_y + 96; y++)
  {
    for (int x = min_x; x < min_x + 96; x++)
    {
      // Convert the pixel to signed 8-bit by subtracting 128 to center the values
      image_data[index++] = static_cast<int8_t>(data[(y * 176) + x] - 128);
    }
  }

  return kTfLiteOk;
}

#endif // ARDUINO_EXCLUDE_CODE
