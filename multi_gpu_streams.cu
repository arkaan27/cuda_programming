

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <stdio.h>

#define MAX_NUM_DEVICES (4)


#define N 1024*1024*8

#define WORK_ITEMS 20

#define ARRAYSIZE N * WORK_ITEMS


cudaStream_t stream[MAX_NUM_DEVICES];

char device_prefix[MAX_NUM_DEVICES][300];

// Declare an array of device poitners to hold input and output array for each device within a multi-GPU system
int *dev_a[MAX_NUM_DEVICES];
int *dev_b[MAX_NUM_DEVICES];
int *dev_c[MAX_NUM_DEVICES];


// Pointers to pinned memory on host that holds the entire array
int *host_a;
int *host_b;
int *host_c;

unsigned *num_processed[MAX_NUM_DEVICES];

cudaEvent_t copy_start_event[MAX_NUM_DEVICES];

cudaEvent_t kernel_start_event[MAX_NUM_DEVICES];

cudaEvent_t kernel_stop_event[MAX_NUM_DEVICES];

cudaEvent_t time_to_finish_event[MAX_NUM_DEVICES];

bool processed_result[MAX_NUM_DEVICES];

float time_copy_to_ms = 0.0f;
float time_kernel_ms = 0.0f;
float time_copy_from_ms = 0.0f;
float time_exec_ms = 0.0f;

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



void fill_array(int *a, int* b)
{
	for (int i = 0; i < ARRAYSIZE; i++)
	{

		a[i] = rand();
		b[i] = rand();
	}


}

bool check_array(char* device_prefix, int* host_a, int* host_b, int* host_c)
{

	for (int i = 0; i < ARRAYSIZE; i++)
	{

		float as = (host_a[i]);
		float bs = (host_b[i]);

		int tmp = host_c[i];

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

		c[idx] =  (as + bs) / 2;
	}
}

void cleanup(cudaStream_t* stream, int device_num, int* host_a, int* host_b, int* host_c)
{
	// Make sure all streams have compelted before freeing device memory and deallocating streams
	checkCuda(cudaStreamSynchronize(stream[device_num]));

	checkCuda(cudaEventElapsedTime(&time_copy_to_ms, copy_start_event[device_num], kernel_start_event[device_num]));

	printf("\n%sCopy to time = %3.6fms", device_prefix[device_num], time_copy_to_ms);


	checkCuda(cudaEventElapsedTime(&time_kernel_ms, kernel_start_event[device_num], kernel_stop_event[device_num]));

	printf(" %sKernel time = %3.6fms", device_prefix[device_num], time_kernel_ms);

	checkCuda(cudaEventElapsedTime(&time_copy_from_ms, kernel_stop_event[device_num], time_to_finish_event[device_num]));

	printf("%sCopy from time = %3.6fms", device_prefix[device_num], time_copy_from_ms);

	checkCuda(cudaStreamDestroy(stream[device_num]));

	checkCuda(cudaFree(dev_a[device_num]));
	checkCuda(cudaFree(dev_b[device_num]));
	checkCuda(cudaFree(dev_c[device_num]));

	if (!check_array(device_prefix[device_num], host_a, host_b, host_c))
	{

		printf("device results don't match with host\n");
	}


}



bool process_result(int device_num)
{
	if (cudaEventQuery(time_to_finish_event[device_num]) == cudaSuccess)
	{
		return true;
	}
	else
	return false;
}

void push_work_into_queue(int device_num, int* host_a, int* host_b, int* host_c, int work_item_counter)
{
	checkCuda(cudaSetDevice(device_num));

	checkCuda(cudaEventRecord(copy_start_event[device_num], stream[device_num]));

	checkCuda(cudaMemcpyAsync((void*)dev_a[device_num], host_a+(N*work_item_counter), N * sizeof(int), cudaMemcpyHostToDevice, stream[device_num]));
	checkCuda(cudaMemcpyAsync((void*)dev_b[device_num], host_b + (N*work_item_counter), N * sizeof(int), cudaMemcpyHostToDevice, stream[device_num]));

	checkCuda(cudaEventRecord(kernel_start_event[device_num], stream[device_num]));
	kernel << <N / 256, 256, 0, stream[device_num] >> > (dev_a[device_num], dev_b[device_num], dev_c[device_num]);

	checkCuda(cudaEventRecord(kernel_stop_event[device_num], stream[device_num]));



	checkCuda(cudaMemcpyAsync((void*)(host_c + (N*work_item_counter)), (void*)dev_c[device_num], N * sizeof(int), cudaMemcpyDeviceToHost, stream[device_num]));

	checkCuda(cudaEventRecord(time_to_finish_event[device_num], stream[device_num]));

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void gpu_kernel(int num_devices, int new_work_blocks)
{
	const int shared_memory_usage = 0;


	// Allocate pinned memory on the host
	checkCuda(cudaMallocHost((void**)&host_a, ARRAYSIZE * sizeof(int)));
	checkCuda(cudaMallocHost((void**)&host_b, ARRAYSIZE * sizeof(int)));
	checkCuda(cudaMallocHost((void**)&host_c, ARRAYSIZE * sizeof(int)));


	fill_array(host_a, host_b);

	unsigned results_to_process = num_devices;

	unsigned sleep_count = 0;


	int work_item_counter = 0;

	for (int device_num = 0; device_num < num_devices; device_num++)
	{
		checkCuda(cudaSetDevice(device_num));

		checkCuda(cudaEventCreate(&copy_start_event[device_num]));

		checkCuda(cudaEventCreate(&kernel_start_event[device_num]));

		checkCuda(cudaEventCreate(&kernel_stop_event[device_num]));

		checkCuda(cudaEventCreate(&time_to_finish_event[device_num]));

		struct cudaDeviceProp device_prop;

		checkCuda(cudaGetDeviceProperties(&device_prop, device_num));

		sprintf(&device_prefix[device_num][0], "\nID:%d %s:", device_num, device_prop.name);

		checkCuda(cudaStreamCreate(&stream[device_num]));

		checkCuda(cudaMalloc((void**)&dev_a[device_num], N*sizeof(int)));
		checkCuda(cudaMalloc((void**)&dev_b[device_num], N*sizeof(int)));
		checkCuda(cudaMalloc((void**)&dev_c[device_num], N * sizeof(int)));

		push_work_into_queue(device_num, host_a, host_b, host_c, work_item_counter);

		work_item_counter++;

	}

	 results_to_process = num_devices + new_work_blocks;

	unsigned results_being_calculated = num_devices;


	// While loop queries wrok item completion event to see whether it can assign a new work item to a device, otherwise it will put the thread to sleep
	// for 100 milliseconds
	while (results_to_process != 0)
	{
		for (int device_num = 0; device_num < num_devices; device_num++)
		{

			bool processed_a_result = false;

			if (processed_result[device_num] == false)
			{
				processed_result[device_num] = process_result(device_num);

				if (processed_result[device_num] == true)
				{
					results_to_process--;

					num_processed[device_num]++;

					results_being_calculated--;

					processed_a_result = true;

					// Print the current time on the host to process work items
					printf("%sHost wait time: %u ms\n", device_prefix[device_num], sleep_count * 100);

					if (results_to_process > results_being_calculated)
					{
						push_work_into_queue(device_num, host_a, host_b, host_c, work_item_counter);

						processed_result[device_num] = false;

						results_being_calculated++;

						printf("\nWaiting");
					}

					fflush(stdout);

					work_item_counter++;
				}
				else
				{
					printf(".");
					fflush(stdout);
				}
			}
			sleep_count++;
			Sleep(10);
		}


	}
	for (int device_num = 0; device_num < num_devices;device_num++)
	{


		cleanup(stream, device_num, host_a, host_b, host_c);
	}


	// Free host memory
	checkCuda(cudaFreeHost(host_a));
	checkCuda(cudaFreeHost(host_b));
	checkCuda(cudaFreeHost(host_c));
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCuda(cudaDeviceReset());

}

int main()
{
	int num_devices;

	checkCuda(cudaGetDeviceCount(&num_devices));

	checkCuda(cudaSetDeviceFlags(cudaDeviceScheduleYield));

	if (num_devices > MAX_NUM_DEVICES)
	{
		num_devices = MAX_NUM_DEVICES;
	}

	int num_work_items = WORK_ITEMS - num_devices;

	gpu_kernel(num_devices, num_work_items);



	return 0;
}