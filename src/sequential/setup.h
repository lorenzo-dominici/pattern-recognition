#ifndef SETUP_H
#define SETUP_H

#include "../model/dataset.h"

// Function to define results based on the knowledge base and queries
// void define_results(dataset_t* kb, dataset_t* queries, dataset_t* results);

// Function to define times for a given number of runs based on the queries
void define_times(dataset_t* queries, unsigned int n_runs, dataset_t* times);

#endif