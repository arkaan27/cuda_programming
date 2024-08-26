
#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>


#define SIZE 65536

float h_a[SIZE];
float h_b[SIZE];

#define BLOCK_SIZE 256
#define NUMBER_OF_BLOCKS 64

cudaError_t cudaStatus;

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




 __device__ void reduceBlock(float* shared_data)
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



}

__global__ void DotProduct(float* a, float* b, float* d_out, int size)
{
	__shared__ float shared_data[BLOCK_SIZE];

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int tid = threadIdx.x;

	float temp = 0.0f;

	// Multiply together each dot product pair and move onto the next pair
	while (index < size)
	{
		temp += a[index] * b[index];
		index += blockDim.x * gridDim.x;

	}

	shared_data[tid] = temp;

	// Perform parallel reduction for block
	reduceBlock(shared_data);

	// First element in shared array contains the partial sum for the threadblock, so first thread stores in output array
	if (tid == 0)
	{

		d_out[blockIdx.x] = shared_data[0];
	}

}

__global__ void Reduction(float* d_out, int size)
{
	__shared__ float shared_data[NUMBER_OF_BLOCKS];

	int tid = threadIdx.x;

	if (tid < size)
	{

		shared_data[tid] = d_out[tid];

		reduceBlock(shared_data);

		// First element in shared array contains the partial sum for the threadblock, so first thread stores in output array
		if (tid == 0)
		{

			d_out[0] = shared_data[0];
		}
	}
}


float* InitData(float* a, float initVal, int size)
{

	for (int i = 0; i < size; i++)
	{

		a[i] = initVal;
	}
	return a;
}

bool checkResult(float* a, float* b, float h_result,float& cpu_result,  int size)
{

	float temp = 0.0f;

	for (int i = 0; i < size; i++)
	{
		temp += a[i] * b[i];

	}

	cpu_result = temp;

	float diff = (abs(temp - h_result) / temp) * 100;


	// Check whether difference between host and device is within 0.1% difference
	if (diff > 0.1f)
	{
		return false;
	}

	return true;

}


int CalculateNumberOfBlocks(int arraySize, int threadsPerBlock)
{
	int num_blocks = NUMBER_OF_BLOCKS;
	if (arraySize < (NUMBER_OF_BLOCKS * threadsPerBlock))
	{
		num_blocks = (arraySize + (threadsPerBlock - 1)) / threadsPerBlock;
	}

	return num_blocks;
}


int main()
{

	// varaible to hold the final product result calculated on the device
	float h_result = 0.0f;

	// variable for checking results on host
	float cpu_result = 0.0f;

	// Initialize input vectors with arbritrary values
	InitData(h_a, 0.33f, SIZE);
	InitData(h_b, 0.66f, SIZE);

	// Create pointers to point to memory on device
	float* d_a, *d_b, *d_out;

	// Allocate GPU buffers for three vectors (two input, one output)
	checkCuda(cudaMalloc((void**)&d_a, SIZE * sizeof(float)));
	checkCuda(cudaMalloc((void**)&d_b, SIZE * sizeof(float)));
	checkCuda(cudaMalloc((void**)&d_out, NUMBER_OF_BLOCKS * sizeof(float)));

	// Copy input vectors from host memory to GPU buffers.
	checkCuda(cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice));

	// Calculate Number of Blocks e.g. <= 64
	int num_blocks = CalculateNumberOfBlocks(SIZE, BLOCK_SIZE);

	// Create grid launch parameters
	dim3 GridSize(num_blocks);
	dim3 BlockSize(BLOCK_SIZE);

	// Initialize output array to zero
	checkCuda(cudaMemset(d_out, 0, sizeof(float) * NUMBER_OF_BLOCKS));


	// Main dot product kernel. Will produce one partial sum per block which is the sum of the dot product pairs calculated by the block
	DotProduct << < GridSize, BlockSize >> > (d_a, d_b, d_out, SIZE);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	checkCuda(cudaDeviceSynchronize());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DotProduct launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Launch parallel reduction with one threadblock, with block size = number of blocks launched by the Dot Product Kernel
	Reduction << < 1, GridSize >> > (d_out, SIZE);

	// Synchronize kernel with host
	checkCuda(cudaDeviceSynchronize());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Reduction launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	//Print Result of Dot Product
	checkCuda(cudaMemcpy(&h_result, &d_out[0], sizeof(float), cudaMemcpyDeviceToHost));


	// do check to see whether device results are within error tolerance of cpu results
	bool match = checkResult(h_a, h_b, h_result,cpu_result, SIZE);

	if (match)
	{
		printf("Results verified on host\n");
	}
	else
	{
		printf("Results NOT verified on host\n");
	}

	// Print result obtained from Host and Device and calculate difference due to rounding errors
	printf("Device Result = %3.6f, Host result = %3.6f, Difference = %3.2f%%\n", h_result, cpu_result, (abs(cpu_result - h_result)/ cpu_result)*100);

	//free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCuda(cudaDeviceReset());

	return 0;
}