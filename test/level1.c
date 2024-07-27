#include "cblas.h"
#include "cblas_clinit.h"
#include <stdio.h>

int main() {
    cblas_clinit();
    int N = 100;
    fComplex *A = (fComplex *)malloc(sizeof(fComplex) * N);
    fComplex *B = (fComplex *)malloc(sizeof(fComplex) * N);
    // dComplex *A = (dComplex *)malloc(sizeof(dComplex) * N);
    // dComplex *B = (dComplex *)malloc(sizeof(dComplex) * N);
    // double *A = (double *)malloc(sizeof(double) * N);
    // double *B = (double *)malloc(sizeof(double) * N);
    fComplex alpha = {0.0, 0.0};
    for (int i = 0; i < N; i++) {
        A[i].real = 1.0;
        A[i].imag = 1.0;
        B[i].real = 2.0;
        B[i].imag = 2.0;
        // A[i] = 1.0;
        // B[i] = 2.0;
    }

    cblas_cdotu_sub(N, (void *)A, 2, (void *)B, 2, (void *)&alpha);
    /*
    for (int i = 0; i < N; i++) {
        // printf("%f, %f ---- %f, %f\n", A[i].real, A[i].imag, B[i].real, B[i].imag);
        printf("%f, %f \n", A[i], B[i]);
    }
    */
    printf("%f, %f \n", alpha.real, alpha.imag);
    // printf("res : %f\n", res);
    free(A);
    free(B);
}
