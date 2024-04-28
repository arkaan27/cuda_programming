#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{

    printf("Hello Cuda World \n")
}

int main()
{
    hello_cuda << <1,1 >> > ();
    cudaDeviceSynhronize();

    cudaDeviceReset();
    return 0;

}