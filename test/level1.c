#include "cblas.h"
#include "cblas_clinit.h"
#include <stdio.h>

int main() {
    cblas_clinit();
    int N = 100;
    float *A = (float *)malloc(sizeof(float) * N);
    float *B = (float *)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }
    cblas_sswap(N, A, 2, B, 2);
    for (int i = 0; i < N; i++) {
        printf("%f %f\n", A[i], B[i]);
    }
    free(A);
    free(B);
}
