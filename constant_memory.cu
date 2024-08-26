#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__constant__  float const_vars[16] = { 0.33f, 0.567f, 0.456f, 0.3456f, 0.436f, 0.4356f, 0.2883f, 0.4857845f, 0.458745f, 0.87685f, 0.4578475f, 0.78675f, 0.878567f, 0.454f, 0.758475f, 0.9897f };


template<typename T>
__global__ void ConstantAccess(T* a, int interation_size)
{
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	//a[index] = a[index] + const_vars[blockDim.x];
	if ((blockIdx.x + interation_size) < 16384)
	{

		for (int i = 0; i < 16; i++)
		{

			a[index] = a[index] + const_vars[i];
		}
	}
}


__global__ void GlobalAccess(float* a,   float*  aConst, int interation_size)
{

	int index = (blockDim.x * blockIdx.x + threadIdx.x);// *s;

	if ((blockIdx.x + interation_size) < 16384)
	{

		for (int i = 0; i < 16; i++)
		{

			a[index] = a[index] + aConst[i];
		}

	}
	if (index == 0)
	{
		//printf("a[index] = %3.6f\n", a[index]);

	}


}

int main()
{

	float* h_vars, *h_varsTmp;
	const float h_varsConst[16] = { 0.33f, 0.567f, 0.456f, 0.3456f, 0.436f, 0.4356f, 0.2883f, 0.4857845f, 0.458745f, 0.87685f, 0.4578475f, 0.78675f, 0.878567f, 0.454f, 0.758475f, 0.9897f };
	float* g_vars;
	 float*  g_varsConst;


	int blockSize = 256;

	int iteration_size = 256;
	cudaEvent_t startEvent, stopEvent;

	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));

	int n = 16384*256;

	float nMb = (float)(n / (1024 * 1024));

	float elapsedTime = 0.0f;

	h_vars = (float*)malloc(sizeof(float) * n);
	h_varsTmp = (float*)malloc(sizeof(float) * n);

	//h_varsConst = (float*)malloc(sizeof(float) * 16);

	for (int i = 0; i < 16; i++)
	{


		//printf("h_varsConst[%d] = %3.6f\n", i, h_varsConst[i]);
	}



	HANDLE_ERROR(cudaMalloc((void**)&g_vars, sizeof(float) * n));

	HANDLE_ERROR(cudaMalloc((void**)&g_varsConst, sizeof( float) * 16));

	HANDLE_ERROR(cudaMemcpy(g_vars, h_vars, sizeof(float) * n,
		cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(g_varsConst, h_varsConst, sizeof( float) * 16,
		cudaMemcpyHostToDevice));

	//HANDLE_ERROR(cudaMemcpyToSymbol(const_vars, h_varsConst, sizeof(float) * 16));

	int blockSizeNumThreads = n / blockSize;

	dim3 db(blockSizeNumThreads);
	dim3 dg(blockSize);

	//test constant access
	printf("\n========== Global memory test ==============\n");

	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	GlobalAccess << < db, dg >> > (g_vars, g_varsConst, iteration_size);

	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
	printf("%f Gb/s, total time = %3.6fms\n", 2 * nMb / elapsedTime, elapsedTime);

	HANDLE_ERROR(cudaMemcpy(h_varsTmp, g_vars, sizeof(float) * n,
		cudaMemcpyDeviceToHost));


	//reset test data
	HANDLE_ERROR(cudaMemcpy(g_vars, h_vars, sizeof(float) * n,
		cudaMemcpyHostToDevice));
	printf("\n========== Constant memory test ==============\n");
	// test global access

	HANDLE_ERROR(cudaEventRecord(startEvent, 0));


	ConstantAccess<float> << < db, dg >> > (g_vars, iteration_size);
	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
	printf("%f Gb/s, total time = %3.6fms\n", 2 * nMb / elapsedTime, elapsedTime);

	HANDLE_ERROR(cudaMemcpy(h_vars, g_vars, sizeof(float) * n,
		cudaMemcpyDeviceToHost));

	// Check results

	int loopCount = 0;

	for (int i = 0; i < n, loopCount < 100; i++)
	{

		if (h_vars[i] != h_varsTmp[i])
		{
			printf("Values do not match: h_vars[%d] = %3.6f, h_varsTmp[%d] = %3.6f\n", i, h_vars[i], i, h_varsTmp[i]);
		}

		loopCount++;
	}

	int input = 0;
    std::cin >> input;
}
