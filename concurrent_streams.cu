#include "cuda_runtime.h"


#include <Windows.h>
#include <stdio.h>

#define N 1024*1024
#define ARRAYSIZE N * 20


cudaStream_t stream0;
cudaStream_t stream1;

cudaEvent_t kernel_start_event;
cudaEvent_t kernel_stop_event;

cudaEvent_t start, stop, sync_event0, sync_event1;

float elapsedTime;

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

bool checkArray(int* a, int* b, int* c)
{

	for (int i = 0; i < ARRAYSIZE; i++)
	{

		float as = (a[i]);
		float bs = (b[i]);

		int tmp = c[i];

		if (tmp != (int)((as + bs) / 2))
		{
			return false;
		}

	}
	return true;
}

__global__ void kernel(int*a, int* b, int* c)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N)
	{
		//int idx1 = (idx + 1);// % 256;
		//int idx2 = (idx + 2);// % 256;


		float as = (a[idx]);
		float bs = (b[idx]);

		c[idx] = (as + bs) / 2;
	}
}

cudaError_t  DefaultStreamExecution(int* dev_a0, int* dev_b0, int* dev_c0, int* host_a, int* host_b, int* host_c)
{
	cudaError_t error;
	for (int i = 0; i < ARRAYSIZE; i += N)
	{

		checkCuda(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice));

		kernel << <N / 256, 256 >> > (dev_a0, dev_b0, dev_c0);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("kernel() failed to launch error = %d\n", error);
		}

		checkCuda(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost));

	}



	return error;
}

cudaError_t  ConcurrentStreamExecution(int* dev_a0, int* dev_b0, int* dev_c0, int* dev_a1, int* dev_b1, int* dev_c1, int* host_a, int* host_b, int* host_c)
{
	cudaError_t error;
	for (int i = 0; i < ARRAYSIZE; i += N * 2)
	{

		checkCuda(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		checkCuda(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));

		checkCuda(cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
		checkCuda(cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
		kernel << <N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("kernel() failed to launch error = %d\n", error);
		}


		kernel << <N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("kernel() failed to launch error = %d\n", error);
		}
		checkCuda(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
		checkCuda(cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));

	}


	return error;
}


int main()
{
	cudaError_t error;
	cudaDeviceProp prop;

	int device;

	checkCuda(cudaGetDevice(&device));
	checkCuda(cudaGetDeviceProperties(&prop, device));

	if (!prop.deviceOverlap)
	{
		printf("Device does not handle overlaps\n");
		return 0;
	}



	checkCuda(cudaEventCreate(&kernel_start_event));
	checkCuda(cudaEventCreate(&kernel_stop_event));


	checkCuda(cudaStreamCreate(&stream0));
	checkCuda(cudaStreamCreate(&stream1));

	int *host_a, *host_b, *host_c;
	int *dev_a0, *dev_b0, *dev_c0;
	int *dev_a1, *dev_b1, *dev_c1;

	checkCuda(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
	checkCuda(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
	checkCuda(cudaMalloc((void**)&dev_c0, N * sizeof(int)));

	checkCuda(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
	checkCuda(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
	checkCuda(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

	checkCuda(cudaHostAlloc((void**)&host_a, ARRAYSIZE * sizeof(int), cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&host_b, ARRAYSIZE * sizeof(int), cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&host_c, ARRAYSIZE * sizeof(int), cudaHostAllocDefault));


	for (int i = 0; i < ARRAYSIZE; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}


	checkCuda(cudaEventRecord(kernel_start_event, 0));

	error = DefaultStreamExecution(dev_a0, dev_b0, dev_c0, host_a, host_b, host_c);

	if (error != cudaSuccess) {
		printf("DefaultStreamExecution() failed to launch error = %d\n", error);
	}



	checkCuda(cudaEventRecord(kernel_stop_event, 0));
	checkCuda(cudaEventSynchronize(kernel_stop_event));

	checkCuda(cudaEventElapsedTime(&elapsedTime, kernel_start_event, kernel_stop_event));

	printf("Serial Execution: Time taken: %3.1f ms\n", elapsedTime);

	if (!checkArray(host_a, host_b, host_c))
	{
		printf("Results don't match\n");
	}

	checkCuda(cudaEventRecord(kernel_start_event, 0));



	error = ConcurrentStreamExecution(dev_a0, dev_b0, dev_c0, dev_a1, dev_b1, dev_c1, host_a, host_b, host_c);

	if (error != cudaSuccess) {
		printf("ConcurrentStreamExecution() failed to launch error = %d\n", error);
	}

	checkCuda(cudaStreamSynchronize(stream0));
	checkCuda(cudaStreamSynchronize(stream1));

	checkCuda(cudaEventRecord(kernel_stop_event, 0));
	checkCuda(cudaEventSynchronize(kernel_stop_event));

	checkCuda(cudaEventElapsedTime(&elapsedTime, kernel_start_event, kernel_stop_event));

	printf("Concurrent Execution: Time taken: %3.1f ms\n", elapsedTime);

	checkCuda(cudaEventRecord(kernel_stop_event, 0));
	checkCuda(cudaEventSynchronize(kernel_stop_event));

	checkCuda(cudaEventElapsedTime(&elapsedTime, kernel_start_event, kernel_stop_event));

	if (!checkArray(host_a, host_b, host_c))
	{
		printf("Results don't match\n");
	}

	checkCuda(cudaFreeHost(host_a));
	checkCuda(cudaFreeHost(host_b));
	checkCuda(cudaFreeHost(host_c));
	checkCuda(cudaFree(dev_a0));
	checkCuda(cudaFree(dev_b0));
	checkCuda(cudaFree(dev_c0));
	checkCuda(cudaFree(dev_a1));
	checkCuda(cudaFree(dev_b1));
	checkCuda(cudaFree(dev_c1));

	checkCuda(cudaStreamDestroy(stream0));
	checkCuda(cudaStreamDestroy(stream1));

	return 0;
}