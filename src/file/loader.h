#ifndef LOADER_H
#define LOADER_H

#include "../model/dataset.h"

// Function to load data from a CSV file
dataset_t* load_data(char* filename);

#endif