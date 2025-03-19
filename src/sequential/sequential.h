#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../model/dataset.h"

// Function to run the sequential pattern recognition algorithm
void seq_run(dataset_t* kb, dataset_t* queries, /*dataset_t**/ float* result, float* times);

#endif