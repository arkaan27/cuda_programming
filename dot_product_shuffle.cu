#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>

cudaEvent_t start;
cudaEvent_t stop;

#define FULL_MASK 0xffffffff

#define BLOCK_SIZE 128
#define NUMBER_OF_BLOCKS 256

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
int CheckMemory(T* ptr) {

	if (ptr == NULL) {
		printf("Error. Allocation was unsuccessful. \n");
		return 0;
	}
	else
		return 1;
}

__inline__ __device__
float warpReduceSum(float val, int warpSize) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(FULL_MASK, val, offset);
	return val;
}

__inline__ __device__
float reduceBlockShuffle(float val) {


	static __shared__ float shared[32]; // Shared mem for 32 partial sums

	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val, warpSize);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSum(val, warpSize); //Final reduce within first warp

	return val;
}

__global__ void DotProductShuffle(float* a, float* b, float* dev_out, int size) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int tid = threadIdx.x;

	float temp = 0.0f;

	// Multiply together each dot product pair and move onto the next pair
	while (index < size)
	{
		temp += a[index] * b[index];
		index += blockDim.x * gridDim.x;

	}

	// Perform parallel reduction for block
	temp = reduceBlockShuffle(temp);

	if (tid == 0)
	{
		dev_out[blockIdx.x] = temp;
	}
}




__global__ void ReductionShuffle(float* dev_out, int N)
{

	unsigned int gridSize = blockDim.x * 2 * gridDim.x;

	int tid = threadIdx.x;

	float temp = dev_out[tid];
	if (tid < N)
	{

		temp = reduceBlockShuffle(temp);

		if (tid == 0)
		{
			dev_out[0] = temp;
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

int CalculateNumberOfBlocks(int arraySize)
{
	int num_blocks = NUMBER_OF_BLOCKS;
	if (arraySize < (NUMBER_OF_BLOCKS * BLOCK_SIZE))
	{
		num_blocks = (arraySize + (NUMBER_OF_BLOCKS - 1)) / NUMBER_OF_BLOCKS;
	}

	return num_blocks;
}

void DotProductCalculation(int arraySize, dim3 GridSize, dim3 BlockSize, float* h_a, float* h_b)
{

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCuda(cudaSetDevice(0));


	// Create timing events on the device
	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));

	float h_result = 0.0f;
	float elapsedTime = 0.0f;

	float* dev_a, *dev_b, *dev_out;

	// Allocate GPU buffers for three vectors (two input, one output)
	checkCuda(cudaMalloc((void**)&dev_a, arraySize * sizeof(float)));

	checkCuda(cudaMalloc((void**)&dev_b, arraySize * sizeof(float)));

	checkCuda(cudaMalloc((void**)&dev_out, GridSize.x * sizeof(float)));

	// Copy input vectors from host memory to GPU buffers.
	checkCuda(cudaMemcpy(dev_a, h_a, arraySize * sizeof(float), cudaMemcpyHostToDevice));

	checkCuda(cudaMemcpy(dev_b, h_b, arraySize * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);

	DotProductShuffle << < GridSize, BlockSize >> > (dev_a, dev_b, dev_out, arraySize);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); //wait for the event to be executed!
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time taken for Dot Product (Warp Shuffle) = %3.6fms\n", elapsedTime);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DotProductShuffle launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
   // any errors encountered during the launch.
	checkCuda(cudaDeviceSynchronize());

	cudaEventRecord(start, 0);

	// Launch parallel reduction with one threadblock, with block size = number of blocks launched by the Dot Product Kernel
	ReductionShuffle<< < 1, GridSize >> > (dev_out, arraySize);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); //wait for the event to be executed!
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time taken for Reduction (Warp Shuffle) = %3.6fms\n", elapsedTime);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ReductionShuffle launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	//Print Result of Dot Product
	checkCuda(cudaMemcpy(&h_result, &dev_out[0], sizeof(float), cudaMemcpyDeviceToHost));
	printf("Result = %3.6f\n", h_result);

	//free device memory
	cudaFree(dev_a);
	cudaFree(dev_b);
}

int main()
{

	const int arraySize = 65536;

	float* h_a = (float*)malloc(arraySize * sizeof(float));
	float* h_b = (float*)malloc(arraySize * sizeof(float));

	// Check memory allocated successfully on the host
	if (!CheckMemory<float>(h_a))
	{
		return 0;
	}
	if (!CheckMemory<float>(h_b))
	{
		return 0;
	}


	 cudaError_t cudaStatus;

	 // Initialize input data
	 InitData(h_a, 1.0f, arraySize);
	 InitData(h_b, 0.5f, arraySize);

	 // Calculate Number of Blocks e.g. <= 256
	 int num_blocks = CalculateNumberOfBlocks(arraySize);

	 // Set threadblock size
	 const int threadsPerBlock = BLOCK_SIZE;

	 dim3 GridSize(num_blocks);
	 dim3 BlockSize(threadsPerBlock);


	 //Host function which launches kernels
	 DotProductCalculation(arraySize, GridSize, BlockSize, h_a, h_b);



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//free host memory
	free(h_a);
	free(h_b);



	return 0;
}