#include <iostream>
#include <cuda_runtime.h>

__device__ unsigned long long d_result = 0;

__global__ void addition(unsigned const int* arr, int size) {
    u_int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&d_result, arr[idx]);
    }
}

__global__ void optimizedAddition(unsigned const int* arr, int size) {
    extern __shared__ unsigned int sharedArr[];

    // Version 1: less optimal
//    u_int arr_idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (arr_idx < size) {
//        sharedArr[threadIdx.x] = arr[arr_idx];
//    } else {
//        sharedArr[threadIdx.x] = 0;
//    }

    // Version 2: Optimized to use all threads to do 1st round sum
    // Preemptively do 1st round sum here
    u_int arr_idx = threadIdx.x + blockIdx.x * blockDim.x * 2;

    unsigned int sum = 0;
    if (arr_idx < size) {
        sum = arr[arr_idx];
    }
    if (arr_idx + blockDim.x < size) {
        sum += arr[arr_idx + blockDim.x];
    }

    sharedArr[threadIdx.x] = sum;

    __syncthreads();

    for (unsigned int endIndex = blockDim.x / 2; endIndex > 0; endIndex /= 2) {
        if (threadIdx.x < endIndex) {
            sharedArr[threadIdx.x] += sharedArr[threadIdx.x + endIndex];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&d_result, sharedArr[0]);
    }
}

unsigned long long additionCpu(unsigned const int* arr, int size) {
    unsigned long long sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    constexpr unsigned int BLOCK_SIZE = 256;
    constexpr unsigned long long N = 10'000'000;

    // Malloc for host array
    auto* h_arr = static_cast<unsigned int *>(malloc(sizeof(unsigned int) * N));

    // Initialise host array Elements
    for (int i = 0; i < N; i++) {
        h_arr[i] = 1;
    }

    // Malloc for device array
    unsigned int* d_arr;
    cudaMalloc((void**) &d_arr, N * sizeof(unsigned int));

    // Copy host array to device array
    cudaMemcpy(d_arr, h_arr, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // define block dimension
    const dim3 block_dim = { BLOCK_SIZE, 1, 1 };

    // define grid dimension
    // For non-optimisation and optimisation version 1;
    // const dim3 gridDim = { (N + BLOCK_SIZE * 2 - 1) / (2 * BLOCK_SIZE), 1, 1 };

    // For optimisation version 2
    const dim3 grid_dim = { (N + BLOCK_SIZE * 2 - 1) / (2 * BLOCK_SIZE), 1, 1 };

    // define shared memory size
    const int sharedMemorySize = BLOCK_SIZE * sizeof(unsigned int);

    // set device result initial value
    cudaMemset(&d_result, 0, sizeof(unsigned long long));

    // invoke kernel
    optimizedAddition<<<grid_dim, block_dim, sharedMemorySize>>>(d_arr, N);
    cudaDeviceSynchronize();

    // define host result
    unsigned long long h_result = 0.0f;

    // copy device result to host result
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(unsigned long long));

    // CPU version for comparison
    unsigned long long cpuResult = additionCpu(h_arr, N);

    std::cout << "GPU SUM: " << h_result << std::endl;
    std::cout << "CPU SUM: " << cpuResult << std::endl;

    // Free memory
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
