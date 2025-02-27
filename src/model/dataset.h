#ifndef DATASET_H
#define DATASET_H

// Structure to represent a dataset
typedef struct {
    char ***names;          // 3D array of names
    unsigned int *lengths;  // Array of lengths
    char size;              // Size of the dataset
    void *data;             // Pointer to the data
} dataset_t;

// Function to initialize a dataset
dataset_t init_dataset(char*** names, unsigned int* lengths, char size, void* data);

// Function to combine two datasets into a result dataset
void from_datasets(dataset_t* a, dataset_t* b, dataset_t* result);

// Function to check if a dataset is defined
int is_defined(dataset_t* data);

// Function to free the memory allocated for a dataset
void free_dataset(dataset_t* data);

#endif