// CudaVectorAdditon.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define SIZE 2048
#define BLOCK_SIZE 128

// create three arrays on the host
int h_a[SIZE];
int h_b[SIZE];
int h_c[SIZE];


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



__global__ void addVectorDevice(int* a, int* b, int* c, int n)
{
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)

	{
		c[tid] = a[tid] + b[tid];
	}
}

bool hostVerify(int* a, int* b, int* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		int index = i;

		if (c[index] != a[index] + b[index])
		{
			return false;
		}
	}

	return true;
}


int main()
{
	// flag to indicate any errors returned from device calls
	cudaError_t error;


	// set device to be the primary adapter
	checkCuda(cudaSetDevice(0));
	

	// populate input arrays
	for (int i = 0; i < (SIZE); i++)
	{

		h_a[i] = 2;

	}
	for (int i = 0; i < (SIZE); i++)
	{
		h_b[i] = 3;
	}


	// create device pointers
	int* d_a, *d_b, *d_c;


	// Allocate area of memory on host to pointers we have declared
	checkCuda(cudaMalloc(&d_a, sizeof(int)* SIZE));
	checkCuda(cudaMalloc(&d_b, sizeof(int)* SIZE));
	checkCuda(cudaMalloc(&d_c, sizeof(int)* SIZE));

	// Copy host input arrays to device arrays
	checkCuda(cudaMemcpy(d_a, h_a, sizeof(int)*SIZE, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_b, sizeof(int)*SIZE, cudaMemcpyHostToDevice));

	// reset device output array
	checkCuda(cudaMemset(d_c, 0, sizeof(int)*SIZE));
	
	int num_blocks = (SIZE + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
	// Launch kernel to add the two vectors
	addVectorDevice << <num_blocks, BLOCK_SIZE >>> (d_a, d_b, d_c, SIZE);

	// Make sure kernel has finished operation
	checkCuda(cudaDeviceSynchronize());

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("addVectorDevice() failed to launch error = %d\n", error);
	}

	// Copy results from device back to host
	checkCuda(cudaMemcpy(h_c, d_c, sizeof(int) * SIZE, cudaMemcpyDeviceToHost));

	// do check to see whether device results match cpu results
	bool match = hostVerify(h_a, h_b, h_c, SIZE);

	// Print result of verification on host
	if (match)
	{
		printf("\nRESULTS MATCH\n");
	}
	else
	{
		printf("\nRESULTS DO NOT MATCH\n");
	}

	
	// free memory allocated on device
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));
	checkCuda(cudaFree(d_c));

	
	checkCuda(cudaDeviceReset());
}
