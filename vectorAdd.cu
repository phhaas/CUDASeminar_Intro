#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

const static int N = 10;

// kernel funtion
__global__ 
void addVectors(float* in1, float* in2, float* out) {
    //global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in1[i] + in2[i];
}

int main() {
    float* in1,  * in2,  * out;
    float* Din1, * Din2, * Dout;

    int size = sizeof(float) * N;

    // allocate memory for host
    in1 = (float*)malloc(size);
    in2 = (float*)malloc(size);
    out = (float*)malloc(size);

    // set i-th element to i in in1
    for (int i = 0; i < N; i++) {
        in1[i] = (float)i;
    }
    
    // set i-th element to N-i-1 in in2
    for (int i = 0; i < N; i++) {
        in2[i] = (float)(N - i - 1);
    }

    // allocate memory for device
    cudaMalloc((void**)&Din1, size);
    cudaMalloc((void**)&Din2, size);
    cudaMalloc((void**)&Dout, size);

    // copy input from host to device
    cudaMemcpy(Din1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Din2, in2, size, cudaMemcpyHostToDevice);

    // calculate addition for all entries in parallel
    addVectors << <1, N >> > (Din1, Din2, Dout);

    // wait for GPU
    cudaDeviceSynchronize();

    //copy result from device to host
    cudaMemcpy(out, Dout, size, cudaMemcpyDeviceToHost);

    printf("All entries in out vector should be N-1=%d.\n", N - 1);

    // write vector to output
    printf("out vector:\n");
    for (int i = 0; i < N; i++) {
        printf("\t%1.1f", out[i]);
    }
    printf("\n");

    //free memory
    free(in1);
    free(in2);
    free(out);
    cudaFree(Din1);
    cudaFree(Din2);
    cudaFree(Dout);

    return 0;
}