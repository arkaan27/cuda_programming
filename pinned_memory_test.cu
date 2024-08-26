#include "cuda_runtime.h"
#include <Windows.h>
#include <stdio.h>
#include <assert.h>


#define MBSIZE 1024 * 1024

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
        ///assert(result == cudaSuccess);
    }
#endif
    return result;
}

void MemoryTest(int* host_array, int* dev_array, int n)
{


    checkCuda(cudaMemcpy(dev_array, host_array, n * sizeof(int), cudaMemcpyHostToDevice));


}



int main()
{
    int *host_arrayPaged, *host_arrayPinned;
    int *dev_array;

    float ms;

    int nMB = 256;

    int n = nMB * MBSIZE / sizeof(int);

    cudaEvent_t startEvent, stopEvent;


    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));


    // Declare pinned memory
    checkCuda(cudaHostAlloc((void**)&host_arrayPinned, n * sizeof(int), cudaHostAllocDefault));

    host_arrayPaged = (int*)malloc(n * sizeof(int));

    checkCuda(cudaMalloc((void**)&dev_array, n * sizeof(int)));


    checkCuda(cudaEventRecord(startEvent, 0));

    MemoryTest(host_arrayPinned, dev_array, n);

    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    //printf("%d, %f\n", i, 2 * nMB / ms);
    printf("Pinned memory (Gb/s): %f, total time = %3.6fms\n", 2 * nMB / ms, ms);

    checkCuda(cudaEventRecord(startEvent, 0));

    MemoryTest(host_arrayPaged, dev_array, n);

    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    //printf("%d, %f\n", i, 2 * nMB / ms);
    printf("Paged memory (Gb/s): %f, total time = %3.6fms\n", 2 * nMB / ms, ms);


    return 0;
}