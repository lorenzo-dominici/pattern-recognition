#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "dataset.h"

// Initialize a dataset with given names, lengths, size, and data
dataset_t init_dataset(char*** names, unsigned int* lengths, char size, void* data) {
    dataset_t dataset;
    dataset.names = names;
    dataset.lengths = lengths;
    dataset.size = size;
    dataset.data = data;
    return dataset;
}

// Combine two datasets into a result dataset
void from_datasets(dataset_t* a, dataset_t* b, dataset_t* result) {
    // Allocate memory for the combined names array
    char*** names = (char***)malloc((a->size + b->size - 1) * sizeof(char**));
    if (!names) {
        perror("Failed to allocate memory for names");
        return;
    }

    // Allocate memory for each name array in the combined dataset
    for (int i = 0; i < a->size + b->size - 2; i++) {
        names[i] = (char**)malloc((i < a->size - 1 ? a->lengths[i] : b->lengths[i]) * sizeof(char*));
        if (!names[i]) {
            perror("Failed to allocate memory for names");
            for (int j = 0; j < i; j++) {
                free(names[j]);
            }
            free(names);
            return;
        }
    }
    names[a->size + b->size - 2] = NULL;

    // Allocate memory for the combined lengths array
    unsigned int* lengths = (unsigned int*)malloc((a->size + b->size - 1) * sizeof(unsigned int));
    if (!lengths) {
        perror("Failed to allocate memory for lengths");
        free(names);
        return;
    }

    // Copy names from dataset a to the combined dataset
    for (int i = 0; i < a->size - 1; i++) {
        for (int j = 0; j < a->lengths[i]; j++) {
            names[i][j] = strdup(a->names[i][j]);
        }
    }

    // Copy names from dataset b to the combined dataset
    for (int i = 0; i < b->size - 1; i++) {
        for (int j = 0; j < b->lengths[i]; j++) {
            names[a->size - 1 + i][j] = strdup(b->names[i][j]);
        }
    }

    // Copy lengths from dataset a to the combined dataset
    for (int i = 0; i < a->size - 1; i++) {
        lengths[i] = a->lengths[i];
    }

    // Copy lengths from dataset b to the combined dataset
    for (int i = 0; i < b->size - 1; i++) {
        lengths[a->size - 1 + i] = b->lengths[i];
    }

    // Set the result dataset fields
    result->names = names;
    result->lengths = lengths;
    result->size = a->size + b->size - 1;
    result->data = NULL;
}

// Check if a dataset is defined (all fields are non-NULL)
int is_defined(dataset_t* data) {
    return data->names != NULL && data->lengths != NULL && data->data != NULL;
}

// Free the memory allocated for a dataset
void free_dataset(dataset_t* data) {
    for (int i = 0; i < data->size; i++) {
        if (data->names[i] == NULL) {
            continue;
        }
        for (int j = 0; j < data->lengths[i]; j++) {
            free(data->names[i][j]);
        }
        free(data->names[i]);
    }
    free(data->lengths);
    free(data->names);
    free(data->data);
    free(data);
}