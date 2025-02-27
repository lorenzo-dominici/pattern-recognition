#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "loader.h"

// Function to load data from a CSV file
dataset_t* load_data(char* filename) {
    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Allocate memory for dataset
    dataset_t* data = (dataset_t*)malloc(sizeof(dataset_t));
    if (!data) {
        perror("Failed to allocate memory for dataset_t");
        fclose(file);
        return NULL;
    }
    data->size = 2;

    // Allocate memory for lengths
    data->lengths = (unsigned int*)malloc(data->size * sizeof(unsigned int));
    if (!data->lengths) {
        perror("Failed to allocate memory for lengths");
        free(data);
        fclose(file);
        return NULL;
    }

    // Allocate memory for names
    data->names = (char***)malloc(data->size * sizeof(char**));
    if (!data->names) {
        perror("Failed to allocate memory for names");
        free(data->lengths);
        free(data);
        fclose(file);
        return NULL;
    }
    data->names[1] = NULL;

    // Read the header line to get the number of columns
    char line[1024*1024];
    if (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        data->lengths[0] = 0;
        while (token) {
            data->lengths[0]++;
            token = strtok(NULL, ",");
        }
    }

    // Allocate memory for columns
    data->names[0] = (char**)malloc(data->lengths[0] * sizeof(char*));
    if (!data->names[0]) {
        perror("Failed to allocate memory for columns");
        free(data->names);
        free(data->lengths);
        free(data);
        fclose(file);
        return NULL;
    }

    // Read the header line again to store column names
    fseek(file, 0, SEEK_SET);
    if (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (int i = 0; i < data->lengths[0]; i++) {
            data->names[0][i] = strdup(token);
            token = strtok(NULL, ",");
        }
    }

    // Read the data rows to get the number of rows
    data->lengths[1] = 0;
    while (fgets(line, sizeof(line), file)) {
        data->lengths[1]++;
    }

    // Allocate memory for data
    data->data = (float**)malloc(data->lengths[0] * sizeof(float*));
    if (!data->data) {
        perror("Failed to allocate memory for columns");
        free(data->names[0]);
        free(data->names);
        free(data->lengths);
        free(data);
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < data->lengths[0]; i++) {
        ((float**)(data->data))[i] = (float*)malloc(data->lengths[1] * sizeof(float));
        if (!((float**)(data->data))[i]) {
            perror("Failed to allocate memory for column data");
            for (int j = 0; j < i; j++) {
                free(((float**)(data->data))[j]);
            }
            free(data->data);
            free(data->names[0]);
            free(data->names);
            free(data->lengths);
            free(data);
            fclose(file);
            return NULL;
        }
    }

    // Read the data rows again to store the data
    fseek(file, 0, SEEK_SET);
    fgets(line, sizeof(line), file); // Skip header line
    int j = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (int i = 0; i < data->lengths[0]; i++) {
            ((float**)(data->data))[i][j] = (float)atof(token);
            token = strtok(NULL, ",");
        }
        j++;
    }

    // Close the file
    fclose(file);

    return data;
}