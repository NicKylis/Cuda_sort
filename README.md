# Cuda sorting algorithm
This project features a Bitonic sorting of an array utilizing GPU with CUDA.
Author: Nikolaos Kylintireas
## Summary
This repository contains an implementation of the Bitonic sorting algorithm using CUDA to leverage the parallel processing power of GPUs. The Bitonic sort is particularly well-suited for parallel execution, making it an excellent choice for GPU-based sorting.

## Instructions

### Prerequisites
- CUDA Toolkit installed on your system
- A compatible NVIDIA GPU
- CMake (for building the project)

### Building the Project
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Cuda_sort.git
    cd Cuda_sort
    ```

2. Build the project:
    ```sh
    make <version>
    ```
    version: the version of the code you wish to run. Default: v2 (fastest).
    Type make help for more information

### Running the Code
1. After building the project, you can run the executable:
    ```sh
    ./main <q>
    ```
    q: Defines the size of the array (2 ^ q)
2. The program will sort a predefined array using the Bitonic sorting algorithm on the GPU.

### Customizing the Input
- To sort a custom array, you can modify the input array in the `main.cu` file.

### Troubleshooting
- Ensure that your CUDA environment is correctly set up and that your GPU drivers are up to date.
- Refer to the CUDA Toolkit documentation for additional help.