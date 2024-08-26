#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


typedef  unsigned char byte;

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
void FillArray(byte* hist_data, unsigned N)
{

	for (unsigned i = 0; i < N; i++)
	{
		hist_data[i] = i % 256;

	}
}

__global__ void Histogram(byte* d_hist_data, unsigned int* d_bin_data, unsigned N)
{

	const unsigned tid = (blockIdx.x * (blockDim.x )) + threadIdx.x;


	if (tid < N)
	{

		const unsigned value_u32 = d_hist_data[tid];

		atomicAdd(&(d_bin_data[value_u32]), 1);
	}
}


int main(int argc, char **argv)
{

	cudaError_t error;


	// Fixed number of bins to store the array value count
	const int bin_size = 256;

	// Size of random value array
	const int array_size = 65536;

	// Byte array declared on host to store the array of random bytes
	byte* h_hist_data = (byte*)malloc(sizeof(byte) * array_size);

	// Unsigned array to store the bins
	unsigned int* h_bin_data = (unsigned int*)malloc(sizeof(unsigned int) * bin_size);

	if (!CheckMemory<unsigned int>(h_bin_data))
	{
		return 0;
	}

	// Initialize host array data
	FillArray(h_hist_data, array_size);

	// Device arrays to hold input data and bins
	byte* d_hist_data;
	unsigned int* d_bin_data;

	checkCuda(cudaMalloc(&d_hist_data, sizeof(byte) * array_size));
	checkCuda(cudaMalloc(&d_bin_data, sizeof(unsigned int) * bin_size));

	checkCuda(cudaMemcpy(d_hist_data, h_hist_data, sizeof(byte) * array_size, cudaMemcpyHostToDevice));

	int num_blocks = (array_size + (bin_size - 1)) / bin_size;

	Histogram << <num_blocks, bin_size >> > (d_hist_data, d_bin_data, array_size);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Histogram() failed to launch error = %d\n", error);
	}

	checkCuda(cudaMemcpy(h_bin_data, d_bin_data, sizeof(unsigned int) * bin_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < bin_size; i++)
	{
		printf("h_bin_data[%d] = %d\n", i, h_bin_data[i]);
	}

}