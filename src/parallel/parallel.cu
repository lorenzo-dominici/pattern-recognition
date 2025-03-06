#include <cuda_runtime.h>
#include "../model/dataset.h"

// CUDA kernel to compute SAD
__global__ void sad_kernel(float* d_db, float* d_qs, float* d_rs, int db_x, int db_y, int qs_x) {
    __shared__ float Q[2 * WARP];
    __shared__ float D[blockDim.x];

    int t = threadIdx.x;
    int d = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    float v = 0.0f;

    for (int i = 0; i < TOT; i++) {
        if (t >= QLS) {
            int k = t - QLS;
            Q[k] = (k < qs_x) ? d_qs[k] : 0.0f;
        } else {
            int k = i * OTS + t;
            D[t] = (k < db_y) ? d_db[d * db_y + k] : 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < WPQ; j++) {
            if (t >= DLS) {
                int j1 = j + 1;
                if (j1 < WPQ) {
                    if (t >= QLS) {
                        int k = t - QLS;
                        int h = k + j1 * WARP;
                        Q[k + (j1 % 2) * WARP] = (h < qs_x) ? d_qs[h] : 0.0f;
                    } else {
                        int k = t + j1 * WARP;
                        int h = i * OTS + k;
                        D[(k) % blockDim.x] = (h < db_y) ? d_db[d * db_y + h] : 0.0f;
                    }
                }
            } else {
                int q = (j % 2) * WARP;
                int s = (t + j * WARP) % blockDim.x;
                for (int k = 0; k < WARP; k++) {
                    v += fabsf(D[s + k] - Q[q + k]);
                }
            }

            __syncthreads();
        }

        if (t < DLS && (int k = i * OTS + t) < (db_y - qs_x + 1)) {
            d_rs[k] = v;
        }
    }
}

// Function to run the CUDA kernel
void cuda_run(float* d_db, float* d_qs, float* d_rs, int db_x, int db_y, int qs_x, int qs_y, int threadsPerBlock, int nBlocksX, int nBlocksY, int nBlocksZ) {
    for (int i = 0; i < qs_x; i++) {
        sad_kernel<<<dim3(nBlocksX, nBlocksY, nBlocksZ), threadsPerBlock>>>(d_db, d_qs + i * qs_x, d_rs + i * qs_x * db_x, db_x, db_y, qs_y);
    }
}

// Function to run the parallel pattern recognition algorithm
int par_run(dataset_t* db, dataset_t* queries, dataset_t* results, float* times) {
    int device;
    cudaGetDevice(&device) == cudaSuccess ? 0 : return 1;
    int threadsPerWarp;
    cudaDeviceGetAttribute(&threadsPerWarp, cudaDevAttrWarpSize, device) == cudaSuccess ? 0 : return 1;
    int maxBlocksPerSM;
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, device) == cudaSuccess ? 0 : return 1;
    int maxSharedMemoryPerSM;
    cudaDeviceGetAttribute(&maxSharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device) == cudaSuccess ? 0 : return 1;
    maxSharedMemoryPerSM /= sizeof(float);
    int maxGridDimX;
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, device) == cudaSuccess ? 0 : return 1;
    int maxGridDimY;
    cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, device) == cudaSuccess ? 0 : return 1;
    int maxGridDimZ;
    cudaDeviceGetAttribute(&maxGridDimZ, cudaDevAttrMaxGridDimZ, device) == cudaSuccess ? 0 : return 1;
    
    int maxSharedMemoryPerBlock = ((maxSharedMemoryPerSM / maxBlocksPerSM) / threadsPerWarp) * threadsPerWarp;
    int threadsPerBlock = maxSharedMemoryPerBlock - 2 * threadsPerWarp;

    int nBlocksX = 0, nBlocksY = 0, nBlocksZ = 0;
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

    __constant__ int WARP;
    cudaMemcpyToSymbol(WARP, &threadsPerWarp, sizeof(int)) == cudaSuccess ? 0 : return 1;

    int warpsPerBlock = threadsPerBlock / threadsPerWarp;
    __constant__ int WPB;
    cudaMemcpyToSymbol(WPB, &warpsPerBlock, sizeof(int)) == cudaSuccess ? 0 : return 1;

    int warpsPerQuery = (qs_y + threadsPerWarp - 1) / threadsPerWarp;
    __constant__ int WPQ;
    cudaMemcpyToSymbol(WPQ, &warpsPerQuery, sizeof(int)) == cudaSuccess ? 0 : return 1;
    
    int dataLoadersStart = threadsPerBlock - 2 * threadsPerWarp;
    __constant__ int DLS;
    cudaMemcpyToSymbol(DLS, &dataLoadersStart, sizeof(int)) == cudaSuccess ? 0 : return 1;
    
    int queryLoadersStart = threadsPerBlock - threadsPerWarp;
    __constant__ int QLS;
    cudaMemcpyToSymbol(QLS, &queryLoadersStart, sizeof(int)) == cudaSuccess ? 0 : return 1;
    
    int totalOutputTiles = (db_y - qs_y + dataLoadersStart) / dataLoadersStart;
    __constant__ int TOT;
    cudaMemcpyToSymbol(TOT, &totalOutputTiles, sizeof(int)) == cudaSuccess ? 0 : return 1;
    
    unsigned int db_x = db->lengths[0];
    unsigned int db_y = db->lengths[1];
    unsigned int qs_x = queries->lengths[0];
    unsigned int qs_y = queries->lengths[1];
    unsigned int rs_z = results->lengths[2];

    // Allocate device memory
    float *d_db, *d_qs, *d_rs;
    cudaMalloc((void**)&d_db, db_x * db_y * sizeof(float)) == cudaSuccess ? 0 : return 1;
    cudaMalloc((void**)&d_qs, qs_x * qs_y * sizeof(float)) == cudaSuccess ? 0 : return 1;
    cudaMalloc((void**)&d_rs, qs_x * db_x * rs_z * sizeof(float)) == cudaSuccess ? 0 : return 1;

    // Copy data to device
    for (int i = 0; i < db_x; i++) {
        cudaMemcpy(d_db + i * db_x, db->data + i * db_x, db_y * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess ? 0 : return 1;
    }
    for (int i = 0; i < qs_x; i++) {
        cudaMemcpy(d_qs + i * qs_x, queries->data + i * qs_x, qs_y * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess ? 0 : return 1;
    }

    // Run the kernel multiple times
    for (int i = 0; i < n_runs; i++) {
        cuda_run(d_db, d_qs, d_rs, db_x, db_y, qs_x, qs_y, threadsPerBlock, nBlocksX, nBlocksY, nBlocksZ);
        cudaDeviceSynchronize() == cudaSuccess ? 0 : return 1;
    }

    // Copy results back to host
    for (int i = 0; i < qs_x; i++) {
        for (int j = 0; j < db_x; j++) {
            cudaMemcpy(results->data + i * qs_x * db_x + j * db_x, d_results + i * qs_x * db_x + j * db_x, rs_z * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : return 1;
        }
    }

    // Free device memory
    cudaFree(d_db) == cudaSuccess ? 0 : return 1;
    cudaFree(d_qs) == cudaSuccess ? 0 : return 1;
    cudaFree(d_rs) == cudaSuccess ? 0 : return 1;

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
    par_run(db, queries, results, ((float**)(times->data))[i], n_runs) == 0 ? 0 : printf("Failed to run parallel algorithm\n");

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