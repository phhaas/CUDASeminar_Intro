#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

// run matrix multiplication sequentially 

const static int N = 100;

// kernel function
__global__
void multMatrices(int* in1, int* in2, int* out, const int col, const int row) {
    int tmp = 0;
    for (int n = 0; n < N; n++) {
        tmp += in1[row * N + n] * in2[n * N + col];
    }
    out[row * N + col] = tmp;
}

int main() {
    int* in1, * in2, * out;
    int* Din1, * Din2, * Dout;

    int size = sizeof(float) * N * N;

    // allocate memory for host
    in1 = (int*)malloc(size);
    in2 = (int*)malloc(size);
    out = (int*)malloc(size);

    // assign random elemtens to both matrices
    for (int i = 0; i < N * N; i++) {
        in1[i] = rand() % 10;
    }
    for (int i = 0; i < N * N; i++) {
        in2[i] = rand() % 10;
    }

    // allocate memory for device
    cudaMalloc((void**)&Din1, size);
    cudaMalloc((void**)&Din2, size);
    cudaMalloc((void**)&Dout, size);

    // copy input from host to device
    cudaMemcpy(Din1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Din2, in2, size, cudaMemcpyHostToDevice);

    // iterate over all rows
    for (int row = 0; row < N; row++) {
        // iterate over all columns
        for (int col = 0; col < N; col++) {
            // calculate one element at a time
            multMatrices << <1, 1 >> > (Din1, Din2, Dout, col, row);
        }
    }

    // wait for GPU
    cudaDeviceSynchronize();

    //copy result from device to host
    cudaMemcpy(out, Dout, size, cudaMemcpyDeviceToHost);

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            printf("\t%d", out[n * N + m]);
        }
        printf("\n");
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