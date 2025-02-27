#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "sequential.h"
#include "../model/dataset.h"
#include "../file/loader.h"
#include "../file/dumper.h"
#include "../setup.h"
#include "../sad.h"

// Function to run the sequential algorithm
void seq_run(dataset_t* db, dataset_t* queries, dataset_t* results, float* times) {
    float avg_time = 0;
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    printf("ETA: estimating...\n");
    
    // Loop through each query
    for (int i = 0; i < queries->lengths[0]; i++) {
        QueryPerformanceCounter(&start);
        
        // Loop through each database entry
        for (int j = 0; j < db->lengths[0]; j++) {
            // Perform the SAD operation
            seq_sad(((float**)(db->data))[j], ((float**)(queries->data))[i], db->lengths[1], queries->lengths[1], ((float***)(results->data))[i][j]);
        }
        
        QueryPerformanceCounter(&end);
        times[i] = (float)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
        
        // Calculate average time
        if (i == 0) {
            avg_time = times[i];
        } else {
            avg_time = (avg_time * i + times[i]) / (i + 1);
        }
        
        // Estimate time remaining
        float eta = avg_time * (queries->lengths[0] - i - 1);
        printf("\033[F\033[2K\rETA: %.2f seconds\n", eta);
    }
    
    // Calculate total elapsed time
    float elapsed = 0.0;
    for (int i = 0; i < queries->lengths[0]; i++) {
        elapsed += times[i];
    }
    printf("\033[F\033[2K\rElapsed time: %.6f seconds\n", elapsed);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        perror("Required arguments: <n_runs>");
        return 1;
    }
    int n_runs = atoi(argv[1]);
    printf("Initialization...\n");

    // Load database
    printf("Loading data from %s...\n", "data/db.csv");
    dataset_t* db = load_data("data/db.csv");
    if (!db) {
        perror("Failed to load data");
        return 1;
    }
    printf("Loaded %d elements with %d values\n", db->lengths[0], db->lengths[1]);

    // Load queries
    printf("Loading data from %s...\n", "data/queries.csv");
    dataset_t* queries = load_data("data/queries.csv");
    if (!queries) {
        perror("Failed to load queries");
        free_dataset(db);
        return 1;
    }
    printf("Loaded %d elements with %d values\n", queries->lengths[0], queries->lengths[1]);

    // Define results dataset
    printf("Defining results...\n");
    dataset_t* results = (dataset_t*)malloc(sizeof(dataset_t));
    if (!results) {
        perror("Failed to allocate memory for results");
        free_dataset(queries);
        free_dataset(db);
        return 1;
    }
    define_results(queries, db, results);
    if (!is_defined(results)) {
        perror("Failed to define results");
        free_dataset(queries);
        free_dataset(db);
        return 1;
    }
    printf("Results defined with dimensions %d x %d x %d\n", results->lengths[0], results->lengths[1], results->lengths[2]);

    // Define times dataset
    printf("Defining times...\n");
    dataset_t* times = (dataset_t*)malloc(sizeof(dataset_t));
    if (!times) {
        perror("Failed to allocate memory for times");
        free_dataset(results);
        free_dataset(queries);
        free_dataset(db);
        return 1;
    }
    define_times(queries, n_runs, times);
    if (!is_defined(times)) {
        perror("Failed to define times");
        free_dataset(results);
        free_dataset(queries);
        free_dataset(db);
        return 1;
    }
    printf("Times defined with dimensions %d x %d\n", times->lengths[0], times->lengths[1]);

    // Run the sequential algorithm multiple times
    printf("Running %d times...\n", n_runs);
    for (int i = 0; i < n_runs; i++) {
        seq_run(db, queries, results, ((float**)(times->data))[i]);
    }

    // Dump timing data to file
    printf("Dumping timing data to %s...\n", "data/seq_times.csv");
    dump_dataset(times, "data/seq_times.csv");

    // Clean up
    printf("Cleaning...\n");
    free_dataset(times);
    free_dataset(results);
    free_dataset(queries);
    free_dataset(db);

    printf("Finished!\n");
    return 0;
}