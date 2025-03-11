#include <stdio.h>
#include <stdlib.h>
#include "dumper.h"

void dump_dataset(dataset_t* data, char* filename) {
    // Check if dataset size is 2
    if (data->size != 2) {
        perror("Dataset size must be 2\n");
        return;
    }

    int axis = 0;
    char header = 1;

    // Determine axis and header presence based on names
    if (data->names[0] == NULL && data->names[1] != NULL) {
        axis = 1;
    }
    if (data->names[0] == NULL && data->names[1] == NULL) {
        header = 0;
    }

    // Open file in append mode if it exists, otherwise create a new file
    FILE *file = fopen(filename, "r");
    if (file) {
        fclose(file);
        file = fopen(filename, "a");
        header = 0;
    } else {
        file = fopen(filename, "w");
    }

    // Check if file was successfully opened
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Write header if necessary
    if (header) {
        for (int i = 0; i < data->lengths[axis]; i++) {
            fprintf(file, "%s", data->names[axis][i]);
            if (i < data->lengths[axis] - 1) {
                fprintf(file, ",");
            }
        }
    }

    // Write data to file based on axis
    if (axis) {
        for (int i = 0; i < data->lengths[0]; i++) {
            for (int j = 0; j < data->lengths[1]; j++) {
                fprintf(file, "%f", ((float**)(data->data))[i][j]);
                if (j < data->lengths[1] - 1) {
                    fprintf(file, ",");
                } else {
                    fprintf(file, "\n");
                }
            }
        }
    } else {
        for (int j = 0; j < data->lengths[1]; j++) {
            for (int i = 0; i < data->lengths[0]; i++) {
                fprintf(file, "%f", ((float**)(data->data))[i][j]);
                if (i < data->lengths[0] - 1) {
                    fprintf(file, ",");
                } else {
                    fprintf(file, "\n");
                }
            }
        }
    }

    // Close the file
    fclose(file);
}