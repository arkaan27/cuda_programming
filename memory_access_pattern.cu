#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define NUM_ITERATIONS 32
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

template<typename T>
__global__ void offset(T* a,  int s, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x + s;

	if (i < N)
	{
		a[i] = a[i] + 1;
	}

}

template<typename T>
__global__ void stride(T* a,  int s, int N)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;

	if (i < N)
	{
		a[i] = a[i] + 1;
	}

}



template<typename T>
void runTest(int deviceId, int nMB)
{
	int blockSize = 256;

	float ms;

	T *d_a;
	T *d_a2;
	cudaEvent_t startEvent, stopEvent;

	int n = nMB * 1024 * 1024 / sizeof(T);

	// NB:  d_a(33*nMB) for stride case
	int arraySize = n * 33;

	checkCuda(cudaMalloc(&d_a, arraySize * sizeof(T)));


	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	printf("Offset, Bandwidth (GB/s):\n");

	int num_blocks = n / blockSize;

	dim3 db(num_blocks);
	dim3 dg(blockSize);


	for (int i = 0; i <= (NUM_ITERATIONS); i ++) {
		checkCuda(cudaMemset(d_a, 0.0, n * sizeof(T)));

		checkCuda(cudaEventRecord(startEvent, 0));
		offset<T> <<< db, dg >>>(d_a, i, arraySize);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("%d, %f, total time = %3.6f\n", i, 2 * nMB / ms, ms);
	}

	printf("\n");




	printf("Stride, Bandwidth (GB/s):\n");


	for (int i = 1; i <= (NUM_ITERATIONS); i++) {
		checkCuda(cudaMemset(d_a, 0.0, n * sizeof(T)));

		checkCuda(cudaEventRecord(startEvent, 0));
		stride<T><<<  db, dg >>>(d_a, i, arraySize);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		//printf("%d, %f\n", i, 2 * nMB / ms);
		printf("%d, %f, total time = %3.6f\n", i, 2 * nMB / ms, ms);
	}


	printf("\n");



	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	cudaFree(d_a);
}

int main(int argc, char **argv)
{
	int nMB = 4;
	int deviceId = 0;

	cudaDeviceProp prop;

	checkCuda(cudaSetDevice(deviceId));
	checkCuda(cudaGetDeviceProperties(&prop, deviceId));
	printf("Device: %s\n", prop.name);
	printf("Transfer size (MB): %d\n", nMB);


	runTest<double>(deviceId, nMB);

}
