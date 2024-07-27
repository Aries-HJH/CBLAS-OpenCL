/*
 * cblas_cswap.c
 *
 * The program is a C interface to cswap.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_cswap( const int N, void *X, const int incX, void *Y,
                       const int incY)
{
   const char *cswap_kernel = "\n" \
                     "__kernel void cswap(__global float* X, __global float* Y, int N, int incX, int incY) { \n" \
                     "  int id = get_global_id(0);\n" \
                     "  float temp;\n" \
                     "  if (id < N) { \n" \
                     "     if (incX == 1 || (id % incX) == 0) { \n" \
                     "        temp = X[id * 2];\n" \
                     "        X[id * 2] = Y[id * 2];\n" \
                     "        Y[id * 2] = temp;\n" \
                     "        temp = X[id * 2 + 1];\n" \
                     "        X[id * 2 + 1] = Y[id * 2 + 1];\n" \
                     "        Y[id * 2 + 1] = temp;\n" \
                     "     } \n" \
                     "  }\n" \
                     "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 0, 1, NULL, cswap_kernel, "cswap");
}
