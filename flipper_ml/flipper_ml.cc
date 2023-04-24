#include <furi.h>

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model/model.h"

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(hello_world_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 2056;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensor failed.");
    return kTfLiteError;
  }

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  if (input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return kTfLiteError;
  }

  // Get the input quantization parameters
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Obtain a pointer to the output tensor.
  TfLiteTensor* output = interpreter.output(0);

  // Get the output quantization parameters
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  // Check if the output is within a small range of the expected output
  float epsilon = 0.05f;

  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    input->data.int8[0] = golden_inputs[i] / input_scale + input_zero_point;
    interpreter.Invoke();
    float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
    if (abs(sin(golden_inputs[i]) - y_pred) > epsilon) {
      MicroPrintf(
          "Difference between predicted and actual y value "
          "is significant.");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

int32_t flipper_ml_app(void* p) {
    UNUSED(p);
    FURI_LOG_I("TEST", "Hello world");
    FURI_LOG_I("TEST", "I'm flipper_ml!");

    tflite::InitializeTarget();
    TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());

    return 0;
}
