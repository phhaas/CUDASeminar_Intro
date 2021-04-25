
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

//#include <stdio.h>

#include <stdio.h>
#include <cuda_runtime.h>

// device kernel
__global__
void helloWorldDevice() {
	printf("Hello world from device %d!\n", threadIdx.x);
}

int main() {
	printf("Hello world from host!\n");

	// run kernel in 3 instances
	helloWorldDevice <<<1, 3>>> ();
}
