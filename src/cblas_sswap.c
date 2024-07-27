/*
 * cblas_sswap.c
 *
 * The program is a C interface to sswap.
 *
 * Written by Xi'an China.  27/7/2024
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_sswap( const int N, float *X, const int incX, float *Y,
                       const int incY)
{
   const char *sswap_kernel = "\n" \
                     "__kernel void sswap(__global float* X, __global float* Y, int N, int incX, int incY) { \n" \
                     "  int id = get_global_id(0);\n" \
                     "  float temp;\n" \
                     "  if (id < N) { \n" \
                     "     if (incX == 1 || (id % incX) == 0) { \n" \
                     "        temp = X[id];\n" \
                     "        X[id] = Y[id];\n" \
                     "        Y[id] = temp;\n" \
                     "     } \n" \
                     "  }\n" \
                     "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 0, 0, NULL, sswap_kernel, "sswap");
}

