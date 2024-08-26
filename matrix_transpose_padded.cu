#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <assert.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

using namespace std;

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

bool CheckResults(int* h_A, int* h_B, int N)
{
	int err = 0;
	// Check the result and make sure it is correct
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++) {
			err = h_A[ROW * N + COL] - h_B[COL * N + ROW];


			if (err != 0)
			{

				return false;


			}
		}
	}

	return true;
}




__global__ void transposeCoalesced(int *idata, int *odata, int N)
{
	__shared__ int tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;
	if (x < N && y < N) {
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{

			tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

		}
		__syncthreads();

		x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
		y = blockIdx.x * TILE_DIM + threadIdx.y;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

		}
	}
}
__global__ void transposeCoalescedPadded(int *idata, int *odata, int N)
{
	__shared__ int tile[TILE_DIM][TILE_DIM+1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;
	if (x < N && y < N) {
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{

			tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

		}
		__syncthreads();

		x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
		y = blockIdx.x * TILE_DIM + threadIdx.y;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

		}
	}
}


void matrixTranspose(int *A, int *B,  int N) {

	cudaError_t error;
	cudaError_t launchError;

	float msecTotal = 0.0f;

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	checkCuda(cudaEventCreate(&start));


	cudaEvent_t stop;
	checkCuda(cudaEventCreate(&stop));


	int block_size_x = TILE_DIM;
	int block_size_y = BLOCK_ROWS;

	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock;
	threadsPerBlock.x = block_size_x;
	threadsPerBlock.y = block_size_y;
	dim3 blocksPerGrid(1, 1);
	if (N*N > (block_size_x*block_size_y)) {

		blocksPerGrid.x = (N + (block_size_x - 1)) / block_size_x;
		blocksPerGrid.y = (N + (block_size_y - 1)) / block_size_y;

	}


	// Record the start event
	checkCuda(cudaEventRecord(start, NULL));

	transposeCoalescedPadded << <blocksPerGrid, threadsPerBlock >> > (A, B, N);


	// Record the stop event
	checkCuda(cudaEventRecord(stop, NULL));

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	checkCuda(cudaEventSynchronize(stop));
	checkCuda(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("matrixTransposeKernel time= %.3f msec\n", msecTotal);


	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

}

int main()
{

	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	int N = 64;

	int SIZE = N * N;


	// Allocate memory on the host
	int* h_A = (int*)malloc(sizeof(int)*SIZE);
	int* h_B = (int*)malloc(sizeof(int)*SIZE);
	int* h_C = (int*)malloc(sizeof(int)*SIZE);

	if (!CheckMemory<int>(h_A))
	{
		return 0;
	}
	if (!CheckMemory<int>(h_B))
	{
		return 0;
	}
	if (!CheckMemory<int>(h_C))
	{
		return 0;
	}


	// Initialize matrices on the host
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i*N + j] = rand();

		}
	}



	// Allocate memory on the device
	int* d_A;
	int* d_B;


	checkCuda(cudaMalloc(&d_A, sizeof(int)*SIZE));
	checkCuda(cudaMalloc(&d_B, sizeof(int)*SIZE));



	checkCuda(cudaMemcpy(d_A, h_A, sizeof(int)*SIZE, cudaMemcpyHostToDevice));

	//call host matrix function to launch kernel
	matrixTranspose(d_A, d_B, N);

	checkCuda(cudaDeviceSynchronize());


	checkCuda(cudaMemcpy(h_B, d_B, sizeof(int)*SIZE, cudaMemcpyDeviceToHost));
	checkCuda(cudaDeviceSynchronize());



	if (!CheckResults(h_A, h_B, N))
	{

		printf("Calculations failed\n");

	}
	else
	{
		printf("Calculations match\n");
	}


	printf("=============== INPUT MATRIX ========================\n");

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{

			printf("h_A[%-4d] = %-6d\t", i*N +j, h_A[i*N +j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("=============== OUTPUT MATRIX ========================\n");
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{

			printf("h_B[%-4d] = %-6d\t", i*N + j, h_B[i*N + j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("==================================\n");

	cudaFree(d_A);
	cudaFree(d_B);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}