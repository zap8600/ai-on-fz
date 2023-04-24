#include "tensorflow/lite/micro/debug_log.h"

#include <stdio.h>

extern "C" void DebugLog(const char* s) {