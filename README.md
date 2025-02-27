# Pattern Recognition Project

This project is developed for the Parallel Computing course at the University of Florence. The aim is to compare two versions of the Sum of Absolute Differences (SAD)-based pattern recognition algorithm: one sequential version implemented in C, and one parallel version to be implemented in CUDA.

## Project Structure

The project is organized into the following directories:

- `src/model/`: Contains the dataset handling code.
- `src/file/`: Contains the file loading and dumping code.
- `src/sequential/`: Contains the sequential implementation of the SAD-based pattern recognition algorithm.
- `src/parallel/`: Intended for the parallel CUDA implementation of the SAD-based pattern recognition algorithm.
- `src/`: Contains other utility functions and setup code.

## Files

- `src/model/dataset.c`: Handles dataset memory allocation and management.
- `src/file/loader.c`: Loads data from files into memory.
- `src/file/dumper.c`: Dumps data from memory to files.
- `src/sequential/sequential.c`: Implements the sequential SAD-based pattern recognition algorithm.
- `src/sequential/sequential.h`: Header file for the sequential implementation.
- `src/sad.c`: Computes the Sum of Absolute Differences (SAD) between a time series and a query.
- `src/sad.h`: Header file for the SAD computation.
- `src/setup.c`: Contains setup functions for defining results and times datasets.

## External Files

The program expects to find the following files in the `/data` directory to work properly.

- `data/db.csv`: Contains the dataset used for pattern recognition.
- `data/queries.csv`: Contains the queries data used for matching against the dataset.

## How to Run

1. **Compile the project**: Use a C compiler to compile the sequential version of the project.

    ```sh
    gcc -o sequential src/sequential/sequential.c src/model/dataset.c src/file/loader.c src/file/dumper.c src/setup.c src/sad.c -lm
    ```

2. **Run the executable**: Execute the compiled program.

    ```sh
    sequential-pr.exe <n_runs>
    ```

## Attention

This version of the code uses Windows-specific instructions and may not be compatible with other operating systems.

## Future Work

- Implement the parallel version of the SAD-based pattern recognition algorithm using CUDA.
- Compare the performance of the sequential and parallel implementations.

## Authors

This project is developed by Lorenzo Dominici, student of the Parallel Computing course at the University of Florence.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
