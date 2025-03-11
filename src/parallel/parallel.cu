#include <cuda_runtime.h>
#include "../model/dataset.h"

#define WARP 32
#define TPB 960
#define WPB TPB / WARP
#define DLS TPB - 2 * WARP
#define QLS TPB - WARP

// CUDA kernel to compute SAD
__global__ void sad_kernel(float* d_db, float* d_qs, float* d_rs, int db_x, int db_y, int qs_x) {
    __shared__ float Q[2 * WARP];
    __shared__ float D[TPB];

    int t = threadIdx.x;
    int d = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    float v = 0.0f;

    int tot = (db_y - qs_x + DLS) / DLS;
    int wpq = (qs_x + WARP - 1) / WARP;

    for (int i = 0; i < tot; i++) {
        if (t >= QLS) {
            int k = t - QLS;
            Q[k] = (k < qs_x) ? d_qs[k] : 0.0f;
        } else {
            int k = i * DLS + t;
            D[t] = (k < db_y) ? d_db[d * db_y + k] : 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < wpq; j++) {
            if (t >= DLS) {
                int j1 = j + 1;
                if (j1 < wpq) {
                    if (t >= QLS) {
                        int k = t - QLS;
                        int h = k + j1 * WARP;
                        Q[k + (j1 % 2) * WARP] = (h < qs_x) ? d_qs[h] : 0.0f;
                    } else {
                        int k = t + j1 * WARP;
                        int h = i * DLS + k;
                        D[(k) % TPB] = (h < db_y) ? d_db[d * db_y + h] : 0.0f;
                    }
                }
            } else {
                int q = (j % 2) * WARP;
                int s = (t + j * WARP) % TPB;
                for (int k = 0; k < WARP; k++) {
                    v += fabsf(D[s + k] - Q[q + k]);
                }
            }

            __syncthreads();
        }

        int k = i * DLS + t;
        if ((t < DLS) && (k < (db_y - qs_x + 1))) {
            d_rs[k] = v;
        }
    }
}

// Function to run the CUDA kernel
void cuda_run(float* d_db, float* d_qs, float* d_rs, int db_x, int db_y, int qs_x, int qs_y, int threadsPerBlock, int nBlocksX, int nBlocksY, int nBlocksZ, float* times) {
    for (int i = 0; i < qs_x; i++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        sad_kernel<<<dim3(nBlocksX, nBlocksY, nBlocksZ), threadsPerBlock>>>(d_db, d_qs + i * qs_x, d_rs + i * qs_x * db_x, db_x, db_y, qs_y);
        gettimeofday(&end, NULL);
        times[i] = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    }
}

// Function to run the parallel pattern recognition algorithm
int par_run(dataset_t* db, dataset_t* queries, dataset_t* results, dataset_t* times, int n_runs) {

    cudaError_t err;

    unsigned int db_x = db->lengths[0];
    unsigned int db_y = db->lengths[1];
    unsigned int qs_x = queries->lengths[0];
    unsigned int qs_y = queries->lengths[1];
    unsigned int rs_z = results->lengths[2];

    printf("========CUDA========\n");
    int device;
    if ((err = cudaGetDevice(&device)) != cudaSuccess)  { printf("DEBUG A %d\n", err); return 1; }
    printf("Using device %d\n", device);
    int threadsPerWarp;
    if ((err = cudaDeviceGetAttribute(&threadsPerWarp, cudaDevAttrWarpSize, device)) != cudaSuccess) { printf("DEBUG B %d\n", err); return 1; }
    printf("Threads per warp: %d\n", threadsPerWarp);
    int maxBlocksPerSM;
    if ((err = cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, device)) != cudaSuccess) { printf("DEBUG C %d\n", err); return 1; }
    printf("Max blocks per SM: %d\n", maxBlocksPerSM);
    int maxSharedMemoryPerSM;
    if ((err = cudaDeviceGetAttribute(&maxSharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device)) != cudaSuccess) { printf("DEBUG D %d\n", err); return 1; }
    maxSharedMemoryPerSM /= sizeof(float);
    printf("Max shared memory per SM: %d\n", maxSharedMemoryPerSM);
    int maxGridDimX;
    if ((err = cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, device)) != cudaSuccess) { printf("DEBUG E %d\n", err); return 1; }
    printf("Max grid dim X: %d\n", maxGridDimX);
    int maxGridDimY;
    if ((err = cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, device)) != cudaSuccess) { printf("DEBUG F %d\n", err); return 1; }
    printf("Max grid dim Y: %d\n", maxGridDimY);
    int maxGridDimZ;
    if ((err = cudaDeviceGetAttribute(&maxGridDimZ, cudaDevAttrMaxGridDimZ, device)) != cudaSuccess) { printf("DEBUG G %d\n", err); return 1; }
    printf("Max grid dim Z: %d\n", maxGridDimZ);
    int maxThreadsPerBlock;
    if ((err = cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device)) != cudaSuccess) { printf("DEBUG H %d\n", err); return 1; }
    printf("Max threads per block: %d\n", maxThreadsPerBlock);
    int maxThreadsPerMultiprocessor;
    if ((err = cudaDeviceGetAttribute(&maxThreadsPerMultiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device)) != cudaSuccess) { printf("DEBUG I %d\n", err); return 1;}
    printf("Max threads per multiprocessor: %d\n", maxThreadsPerMultiprocessor);
    int maxSharedMemoryPerBlock;
    if ((err = cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device)) != cudaSuccess) { printf("DEBUG J %d\n", err); return 1; }
    maxSharedMemoryPerBlock /= sizeof(float);
    printf("Max shared memory per block: %d\n", maxSharedMemoryPerBlock);
    printf("Max shared memory per SM / Max blocks per SM: %d\n", maxSharedMemoryPerSM / maxBlocksPerSM);
    printf("====================")

    int nBlocksX = 1, nBlocksY = 1, nBlocksZ = 1;
    if (db_x <= maxGridDimX) {
        nBlocksX = db_x;
    } else {
        nBlocksX = maxGridDimX;
        if (db_x <= maxGridDimX * maxGridDimY) {
            nBlocksY = (db_x + maxGridDimX - 1) / maxGridDimX;
        } else {
            nBlocksY = maxGridDimY;
            if (db_x <= maxGridDimX * maxGridDimY * maxGridDimZ) {
                nBlocksZ = (db_x + maxGridDimX * maxGridDimY - 1) / (maxGridDimX * maxGridDimY);
            } else {
                nBlocksZ = maxGridDimZ;
            }
        }
    }
    printf("Blocks per dimension: %d x %d x %d\n", nBlocksX, nBlocksY, nBlocksZ);

    // Allocate device memory
    float *d_db, *d_qs, *d_rs;
    if ((err = cudaMalloc((void**)&d_db, db_x * db_y * sizeof(float))) != cudaSuccess) { printf("DEBUG O %d\n", err); return 1; }
    if ((err = cudaMalloc((void**)&d_qs, qs_x * qs_y * sizeof(float))) != cudaSuccess) { printf("DEBUG P %d\n", err); return 1; }
    if ((err = cudaMalloc((void**)&d_rs, qs_x * db_x * rs_z * sizeof(float))) != cudaSuccess) { printf("DEBUG Q %d\n", err); return 1; }

    // Copy data to device
    for (int i = 0; i < db_x; i++) {
        if ((err = cudaMemcpy(d_db + i * db_y, ((float**)db->data)[i], db_y * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { printf("DEBUG R %d\n", err); return 1; }
    }
    for (int i = 0; i < qs_x; i++) {
        if ((err = cudaMemcpy(d_qs + i * qs_y, ((float**)queries->data)[i], qs_y * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { printf("DEBUG S %d\n", err); return 1; }
    }

    // Run the kernel multiple times
    for (int i = 0; i < n_runs; i++) {
        //cuda_run(d_db, d_qs, d_rs, db_x, db_y, qs_x, qs_y, threadsPerBlock, nBlocksX, nBlocksY, nBlocksZ);
        cuda_run(d_db, d_qs, d_rs, db_x, db_y, qs_x, qs_y, TPB, nBlocksX, nBlocksY, nBlocksZ, ((float**)times->data)[i]);
        if ((err = cudaDeviceSynchronize()) != cudaSuccess) { printf("DEBUG T %d\n", err); return 1; }
    }

    // Copy results back to host
    for (int i = 0; i < qs_x; i++) {
        for (int j = 0; j < db_x; j++) {
            if ((err = cudaMemcpy(((float***)results->data)[i][j], d_rs + i * qs_x * db_x + j * db_x, rs_z * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) { printf("DEBUG U %d\n", err); return 1; }
        }
    }

    // Free device memory
    if ((err = cudaFree(d_db)) != cudaSuccess) { printf("DEBUG V %d\n", err); return 1; }
    if ((err = cudaFree(d_qs)) != cudaSuccess) { printf("DEBUG W %d\n", err); return 1; }
    if ((err = cudaFree(d_rs)) != cudaSuccess) { printf("DEBUG X %d\n", err); return 1; }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        perror("Required arguments: <n_runs>");
        return 1;
    }
    int n_runs = atoi(argv[1]);
    printf("Initialization...\n");

    // Load database
    printf("Loading data from %s...\n", "db.csv");
    dataset_t* db = load_data("db.csv");
    if (!db) {
        perror("Failed to load data");
        return 1;
    }
    printf("Loaded %d elements with %d values\n", db->lengths[0], db->lengths[1]);

    // Load queries
    printf("Loading data from %s...\n", "queries.csv");
    dataset_t* queries = load_data("queries.csv");
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
    par_run(db, queries, results, times, n_runs) == 0 ? 0 : printf("Failed to run parallel algorithm\n");

    // Dump timing data to file
    printf("Dumping timing data to %s...\n", "par_times.csv");
    dump_dataset(times, "par_times.csv");

    // Clean up
    printf("Cleaning...\n");
    free_dataset(times);
    free_dataset(results);
    free_dataset(queries);
    free_dataset(db);

    printf("Finished!\n");
    return 0;
}