#include "cblas.h"
#include "cblas_clinit.h"
#include <stdio.h>

int main() {
    cblas_clinit();
    int N = 100;
    // fComplex *A = (fComplex *)malloc(sizeof(fComplex) * N);
    // fComplex *B = (fComplex *)malloc(sizeof(fComplex) * N);
    dComplex *A = (dComplex *)malloc(sizeof(dComplex) * N);
    dComplex *B = (dComplex *)malloc(sizeof(dComplex) * N);
    fComplex alpha = {0.5, 0.5};
    for (int i = 0; i < N; i++) {
        A[i].real = 1.0;
        A[i].imag = 1.0;
        B[i].real = 2.0;
        B[i].imag = 2.0;
    }
    cblas_zswap(N, A, 2, B, 2);
    for (int i = 0; i < N; i++) {
        printf("%f, %f ---- %f, %f\n", A[i].real, A[i].imag, B[i].real, B[i].imag);
    }
    free(A);
    free(B);
}
