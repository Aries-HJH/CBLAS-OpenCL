/*
 * cblas_caxpy.c
 *
 * The program is a C interface to caxpy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_caxpy( const int N, const void *alpha, const void *X,
                       const int incX, void *Y, const int incY)
{
   const char *caxpy_kernel = "\n" \
                  "__kernel void caxpy(__global float* X, __global float* Y, int N, int incX, int incY, float alpha1, float alpha2) { \n" \
                  "  int id = get_global_id(0);\n" \
                  "  if (id < N) { \n" \
                  "     if (incX == 1 || (id % incX) == 0) { \n" \
                  "        Y[id * 2] = X[id * 2] * alpha1 + Y[id * 2];\n" \
                  "        Y[id * 2 + 1] = X[id * 2 + 1] * alpha2 + Y[id * 2 + 1];\n" \
                  "     } \n" \
                  "  }\n" \
                  "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 0, 1, alpha, caxpy_kernel, "caxpy");
} 
