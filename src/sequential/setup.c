#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "setup.h"

// Function to define results dataset based on queries and db datasets
/*
void define_results(dataset_t* queries, dataset_t* db, dataset_t* results) {
    // Check if the size of queries and db datasets is 2
    if (queries->size != 2 || db->size != 2) {
        perror("Invalid dataset size");
        return;
    }

    // Initialize results dataset from queries and db datasets
    from_datasets(queries, db, results);

    // Calculate the length of the third dimension of results dataset
    results->lengths[2] = db->lengths[1] - queries->lengths[1] + 1;

    // Allocate memory for the first dimension of results dataset
    results->data = (float***)malloc(results->lengths[0] * sizeof(float**));
    if (!results->data) {
        perror("Failed to allocate memory for results");
        return;
    }

    // Allocate memory for the second dimension of results dataset
    for (int i = 0; i < results->lengths[0]; i++) {
        ((float***)results->data)[i] = (float**)malloc(results->lengths[1] * sizeof(float*));
        if (!((float***)results->data)[i]) {
            perror("Failed to allocate memory for result");
            // Free previously allocated memory in case of failure
            for (int j = 0; j < i; j++) {
                free(((float***)results->data)[j]);
            }
            free(results->data);
            return;
        }
    }

    // Allocate memory for the third dimension of results dataset
    for (int i = 0; i < results->lengths[0]; i++) {
        for (int j = 0; j < results->lengths[1]; j++) {
            ((float***)results->data)[i][j] = (float*)malloc((results->lengths[2]) * sizeof(float));
            if (!((float***)results->data)[i][j]) {
                perror("Failed to allocate memory for result");
                // Free previously allocated memory in case of failure
                for (int k = 0; k < j; k++) {
                    free(((float***)results->data)[i][k]);
                }
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < results->lengths[1]; l++) {
                        free(((float***)results->data)[k][l]);
                    }
                    free(((float***)results->data)[k]);
                }
                free(results->data);
                return;
            }
        }
    }
}
*/

// Function to define times dataset based on queries dataset and number of runs
void define_times(dataset_t* queries, unsigned int n_runs, dataset_t* times) {
    // Check if the size of queries dataset is 2
    if (queries->size != 2) {
        perror("Invalid dataset size");
        return;
    }

    // Set the size of times dataset
    times->size = 2;
    // Allocate memory for lengths array in times dataset
    times->lengths = (unsigned int*)malloc(times->size * sizeof(unsigned int));
    if (!times->lengths) {
        perror("Failed to allocate memory for lengths");
        return;
    }
    // Set the lengths of the dimensions in times dataset
    times->lengths[0] = n_runs;
    times->lengths[1] = queries->lengths[0];

    // Allocate memory for names array in times dataset
    times->names = (char***)malloc(times->size * sizeof(char**));
    if (!times->names) {
        perror("Failed to allocate memory for names");
        free(times->lengths);
        return;
    }

    // Initialize the first dimension of names array to NULL
    times->names[0] = NULL;
    // Allocate memory for the second dimension of names array
    times->names[1] = (char**)malloc(times->lengths[1] * sizeof(char*));
    if (!times->names[1]) {
        perror("Failed to allocate memory for names");
        free(times->names);
        free(times->lengths);
        return;
    }

    // Copy the names from queries dataset to times dataset
    for (int i = 0; i < times->lengths[1]; i++) {
        times->names[1][i] = strdup(queries->names[0][i]);
    }

    // Allocate memory for the data array in times dataset
    times->data = (float**)malloc(times->lengths[0] * sizeof(float*));
    if (!times->data) {
        perror("Failed to allocate memory for times");
        free(times->names[1]);
        free(times->names);
        free(times->lengths);
        return;
    }

    // Allocate memory for each run in the data array
    for (int i = 0; i < times->lengths[0]; i++) {
        ((float**)times->data)[i] = (float*)malloc(times->lengths[1] * sizeof(float));
        if (!((float**)times->data)[i]) {
            perror("Failed to allocate memory for time");
            // Free previously allocated memory in case of failure
            for (int j = 0; j < i; j++) {
                free(((float**)times->data)[j]);
            }
            free(times->data);
            free(times->names[1]);
            free(times->names);
            free(times->lengths);
            return;
        }
    }
}