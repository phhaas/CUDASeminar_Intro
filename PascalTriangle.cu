#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

const static int N = 11;

// kernel funtion
__global__
void calcColumn(int* row, const int rowNmb) {
    //global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp;
    // calculate i-th element for increasing rows
    for (int countRow = 0; countRow < rowNmb; countRow++) {
        if (i == 0) {
            tmp = 1;
        }
        else if (i <= countRow) {
            tmp = tmp + row[i - 1];
        }
        else {
            tmp = 0;
        }

        // wait for other threads to finish before overwriting i-1-th element with the i-th
        __syncthreads();
        row[i] = tmp;
        // wait before all threads have written i-th element
        __syncthreads();
        // 0-th thread writes current row to output
        if (i == 0) {
            for (int j = 0; j < rowNmb; j++) {
                if (row[j] == 0) continue;
                printf("%d\t", row[j]);
            }
            printf("\n");
        }
    }

}

// recursive factorial function for checking result
int fac(const int n) {
    if (n <= 0) return 1;
    return n*fac(n-1);
}

int main() {
    int* Drow;

    int size = sizeof(int) * N;

    // allocate memory for device
    cudaMalloc((void**)&Drow, size);

    // calculate addition for all N entries in parallel
    calcColumn<<<1, N>>> (Drow, N);

    // wait for GPU
    cudaDeviceSynchronize();

    // print check
    printf("Control row:\n");
    for (int i = 0; i < N; i++) {
        printf("%d\t", (int)(fac(N-1) / (float)fac(i) / (float)fac(N-1-i)));
    }

    //free memory
    cudaFree(Drow);

    return 0;
}