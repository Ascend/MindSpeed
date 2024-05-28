#ifndef PLUGGABLEALLOCATOR_DECORATOR_H
#define PLUGGABLEALLOCATOR_DECORATOR_H

#include "Recorder.h"

class Decorator {
public:
    static void memory_recorder_start();
    static void memory_recorder_end();
    static void malloc_recorder_start();
    static void malloc_recorder_end();
    static void precise_match_start();
    static void precise_match_end();
};

#endif
