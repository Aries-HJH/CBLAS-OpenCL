/*
 * cblas_ccopy.c
 *
 * The program is a C interface to ccopy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_ccopy( const int N, const void *X,
                      const int incX, void *Y, const int incY)
{
   const char *ccopy_kernel = "\n" \
                  "__kernel void ccopy(__global float* X, __global float* Y, int N, int incX, int incY) { \n" \
                  "  int id = get_global_id(0);\n" \
                  "  if (id < N) { \n" \
                  "     if (incX == 1 || (id % incX) == 0) { \n" \
                  "        Y[id * 2] = X[id * 2];\n" \
                  "        Y[id * 2 + 1] = X[id * 2 + 1];\n" \
                  "     } \n" \
                  "  }\n" \
                  "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 0, 1, NULL, ccopy_kernel, "ccopy");
}
