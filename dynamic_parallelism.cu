#include "cuda_runtime.h"
#include <stdio.h>

#define RECURSION_DEPTH 10

__device__ int v = 0;

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


__device__ void threadBlockDeviceSynchronize(void) {
	__syncthreads();
	if (threadIdx.x == 0)
		cudaDeviceSynchronize();
	__syncthreads();
}


__global__ void recursiveKernel(int depth) {
	// up to depth 10



	if (depth == 10)
		return;


	// launch kernel on device
	if (threadIdx.x == 0) {
		printf("depth = %d\n", depth);
		// launch kernel on device
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

		recursiveKernel << < 1, 8, 0, s >> > (depth + 1);

		//threadBlockDeviceSynchronize();

		if (depth == RECURSION_DEPTH-1) {

			printf("v = %d\n", v);
		}


		cudaStreamDestroy(s);


	}

	atomicAdd(&v, 1);
}



int main()
{
	//parent_k << <1, 2 >> > ();
	int depth = 0;

	float elapsedTime = 0.0f;

	cudaEvent_t startEvent, stopEvent;

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	checkCuda(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10));
	checkCuda(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 4096));

	checkCuda(cudaEventRecord(startEvent, 0));

	recursiveKernel << <1, 8 >> > (depth);

	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
	printf("total time = %3.6fms\n", elapsedTime);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("recursiveKernel() failed to launch error = %d\n", error);
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCuda(cudaDeviceReset());


	return 0;
}