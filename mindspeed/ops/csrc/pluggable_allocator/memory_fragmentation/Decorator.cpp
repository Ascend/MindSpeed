#include "Decorator.h"

void Decorator::memory_recorder_start() {
    MemoryRecorder::start_record();
}

void Decorator::memory_recorder_end() {
    MemoryRecorder::end_record();
}

void Decorator::malloc_recorder_start() {
    MallocRecorder::start_record();
}

void Decorator::malloc_recorder_end() {
    MallocRecorder::end_record();
}

void Decorator::precise_match_start() {
    is_precise_match = true;
}

void Decorator::precise_match_end() {
    is_precise_match = false;
}