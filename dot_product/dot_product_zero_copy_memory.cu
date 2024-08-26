
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include <cooperative_groups.h>

#define BLOCK_SIZE 128
#define NUMBER_OF_BLOCKS 256

cudaEvent_t start;
cudaEvent_t stop;
cudaError_t cudaStatus;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		///assert(result == cudaSuccess);
		exit(0);
	}
#endif

	return result;
}

template<typename T>
int CheckMemory(T* ptr) {

	if (ptr == NULL) {
		printf("Error. Allocation was unsuccessful. \n");
		return 0;
	}
	else
		return 1;
}

__device__ void reduceBlock( float* shared_data,float* dev_out, int index)
{
	int tid = threadIdx.x;

	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{

		if (tid % (2 * s) == 0)
		{
			shared_data[tid] += shared_data[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{

		dev_out[index] = shared_data[tid];
	}

}



__global__ void DotProduct(float* a, float* b, float* dev_out, int size)
{
	__shared__ float shared_data[BLOCK_SIZE];


	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int tid = threadIdx.x;

	float temp = 0.0f;

	//multiply together each dot product pair and move onto the next pair
	while (index < size)
	{
		temp += a[index] * b[index];
		index += blockDim.x * gridDim.x;

	}

	shared_data[tid] = temp;

	reduceBlock(shared_data, dev_out, blockIdx.x);
}



__global__ void Reduction(float* g_idata, int N)
{
	__shared__ float shared_data[NUMBER_OF_BLOCKS];
	unsigned int gridSize = blockDim.x * 2 * gridDim.x;



	int tid = threadIdx.x;

	if (tid < N)
	{

		shared_data[tid] = g_idata[tid];


		reduceBlock(shared_data, g_idata, blockIdx.x);
	}
}



// Helper function for using CUDA to add vectors in parallel.
void DotProductCuda(float *dev_a, float *dev_b,float* dev_out, const int arraySize, const int num_blocks, const int threadsPerBlock)
{


	// Variable to store output result
	float h_out;

	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));


	cudaError_t cudaStatus;

	float elapsedTime = 0.0f;


	dim3 Db(num_blocks);
	dim3 Dg(threadsPerBlock);

	cudaEventRecord(start, 0);

	DotProduct << < Db, Dg >> > (dev_a, dev_b, dev_out, arraySize);

	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop)); //wait for the event to be executed!
	checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken for DotProduct = %3.6fms\n", elapsedTime);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DotProduct launch failed: %s\n", cudaGetErrorString(cudaStatus));

	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCuda(cudaDeviceSynchronize());


	cudaEventRecord(start, 0);

	Reduction << < 1, num_blocks >> > ( dev_out, arraySize);

	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop)); //wait for the event to be executed!
	checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken for Reduction = %3.6fms\n", elapsedTime);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Reduction launch failed: %s\n", cudaGetErrorString(cudaStatus));

	}


	checkCuda(cudaMemcpy(&h_out, &dev_out[0], sizeof(float), cudaMemcpyDeviceToHost));

	//Print Result of Dot Product
	printf("\nResult = %3.6f\n", h_out);
}

float* InitData(float* a, float initVal, int size)
{

	for (int i = 0; i < size; i++)
	{

		a[i] = initVal;
	}
	return a;
}

int CalculateNumberOfBlocks(int arraySize)
{
	int num_blocks = NUMBER_OF_BLOCKS;
	if (arraySize < (NUMBER_OF_BLOCKS * BLOCK_SIZE))
	{
		num_blocks = (arraySize + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
	}
	return num_blocks;
}


int main()
{


	cudaDeviceProp prop;

	int device;

	checkCuda(cudaGetDevice(&device));
	checkCuda(cudaGetDeviceProperties(&prop, device));

	if (!prop.canMapHostMemory)
	{
		printf("Device cannot map host memory.\n");
		return 0;
	}

	checkCuda(cudaSetDevice(0));
	checkCuda(cudaSetDeviceFlags(cudaDeviceMapHost));


	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));


	const int arraySize = 65536;

	float elapsedTime = 0.0f;
	clock_t start_cpu, end_cpu;
	double cpu_time_used;


	// Declare pointers to host memory
	float* h_a;
	float* h_b;


	// Declare pointers to device memory
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_out = 0;


	checkCuda(cudaHostAlloc((void**)&h_a, arraySize * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCuda(cudaHostAlloc((void**)&h_b, arraySize * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));


	// Declare output array in device using cudaMalloc because only read from device
	checkCuda(cudaMalloc(&dev_out, NUMBER_OF_BLOCKS * sizeof(float)));

	// Retrieve device pointers in gpu memory space which map to host address space
	checkCuda(cudaHostGetDevicePointer(&dev_a, h_a, 0));
	checkCuda(cudaHostGetDevicePointer(&dev_b, h_b, 0));

	if (!CheckMemory<float>(h_a))
	{
		return 0;
	}
	if (!CheckMemory<float>(h_b))
	{
		return 0;
	}



	InitData(h_a, 1.0f, arraySize);
	InitData(h_b, 0.5f, arraySize);

	const int num_blocks = CalculateNumberOfBlocks(arraySize);

	const int threadsPerBlock = BLOCK_SIZE;


	DotProductCuda(dev_a, dev_b, dev_out, arraySize, num_blocks, threadsPerBlock);




	checkCuda(cudaFreeHost(h_a));
	checkCuda(cudaFreeHost(h_b));


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCuda(cudaDeviceReset());



	return 0;
}