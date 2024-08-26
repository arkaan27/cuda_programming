
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
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

template<typename T>
void checkMemory(T* input)
{
	assert(input != NULL);
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

float ReductionHost(float* idata, int N)
{
	float temp = 0.0f;
	for (int i = 0; i < N; i++)
	{
		temp += idata[i];


	}
	return temp;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t DotProductCuda(float *a, float *b, float* h_out, int arraySize, int num_blocks, int threadsPerBlock)
{


	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));



	cudaError_t cudaStatus;

	float elapsedTime = 0.0f;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCuda(cudaSetDevice(0));



	dim3 Db(num_blocks);
	dim3 Dg(threadsPerBlock);

	checkCuda(cudaEventRecord(start, 0));

	DotProduct << < Db, Dg >> > (a, b, h_out, arraySize);

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



	return cudaStatus;
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
	cudaError_t error;
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

	// Set device to use mapped memory
	checkCuda(cudaSetDeviceFlags(cudaDeviceMapHost));


	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));


	const int arraySize = 65536;

	float elapsedTime = 0.0f;

	float* h_a;
	float* h_b;
	float* h_out;


	// Declare Unified memory using cudaMallocManaged
	checkCuda(cudaMallocManaged(&h_a, arraySize * sizeof(float)));
	checkCuda(cudaMallocManaged(&h_b, arraySize * sizeof(float)));
	checkCuda(cudaMallocManaged(&h_out, NUMBER_OF_BLOCKS * sizeof(float)));

	checkMemory<float>(h_a);
	checkMemory<float>(h_b);
	checkMemory<float>(h_out);

	InitData(h_a, 1.0f, arraySize);
	InitData(h_b, 0.5f, arraySize);

	const int num_blocks = CalculateNumberOfBlocks(arraySize);

	const int threadsPerBlock = BLOCK_SIZE;

	dim3 Db(num_blocks);
	dim3 Dg(threadsPerBlock);

	checkCuda(cudaEventRecord(start, 0));

	DotProduct << < Db, Dg >> > (h_a, h_b, h_out, arraySize);

	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop)); //wait for the event to be executed!
	checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken for DotProduct = %3.6fms\n", elapsedTime);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DotProduct launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	checkCuda(cudaDeviceSynchronize());

	h_out[0] = ReductionHost(h_out, NUMBER_OF_BLOCKS);

	//Print Result of Dot Product
	printf("\nResult = %3.6f\n", h_out[0]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCuda(cudaDeviceReset());


	//free(a);
	//free(b);

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_out);

	return 0;
}