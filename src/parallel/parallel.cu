#include <cuda_runtime.h>
#include "../model/dataset.h"

// CUDA kernel to compute SAD
__global__ void sad_kernel(float* d_db, float* d_queries, float* d_results, int db_size, int query_size) {
    // TODO: Implement this
}

// Function to run the parallel pattern recognition algorithm
void par_run(dataset_t* db, dataset_t* queries, dataset_t* results, float* times) {
    unsigned int db_x = db->lengths[0];
    unsigned int db_y = db->lengths[1];
    unsigned int qs_x = queries->lengths[0];
    unsigned int qs_y = queries->lengths[1];
    unsigned int rs_z = results->lengths[2];

    // Allocate device memory
    float *d_db, *d_qs, *d_rs;
    cudaMalloc((void**)&d_db, db_x * db_y * sizeof(float));
    cudaMalloc((void**)&d_qs, qs_x * qs_y * sizeof(float));
    cudaMalloc((void**)&d_rs, qs_x * db_x * rs_z * sizeof(float));

    // Copy data to device
    for (int i = 0; i < db_x; i++) {
        cudaMemcpy(d_db + i * db_x, db->data + i * db_x, db_y * sizeof(float), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < qs_x; i++) {
        cudaMemcpy(d_qs + i * qs_x, queries->data + i * qs_x, qs_y * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    int blockSize = 256; // TODO: Tune this
    int numBlocks = (db_size - query_size + 1 + blockSize - 1) / blockSize; // TODO: Tune this
    for (int i = 0; i < n_runs; i++) {
        computeSADKernel<<<numBlocks, blockSize>>>(d_db, d_queries, d_results, db_size, query_size);
    }

    // Copy results back to host
    for (int i = 0; i < qs_x; i++) {
        for (int j = 0; j < db_x; j++) {
            cudaMemcpy(results->data + i * qs_x * db_x + j * db_x, d_results + i * qs_x * db_x + j * db_x, rs_z * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    // Free device memory
    cudaFree(d_db);
    cudaFree(d_qs);
    cudaFree(d_rs);
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

    // Run the parallel algorithm multiple times
    printf("Running %d times...\n", n_runs);
    par_run(db, queries, results, ((float**)(times->data))[i], n_runs);

    // Dump timing data to file
    printf("Dumping timing data to %s...\n", "data/par_times.csv");
    dump_dataset(times, "data/par_times.csv");

    // Clean up
    printf("Cleaning...\n");
    free_dataset(times);
    free_dataset(results);
    free_dataset(queries);
    free_dataset(db);

    printf("Finished!\n");
    return 0;
}