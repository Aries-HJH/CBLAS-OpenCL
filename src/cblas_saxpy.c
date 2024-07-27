/*
 * cblas_saxpy.c
 *
 * The program is a C interface to saxpy.
 * It calls the fortran wrapper before calling saxpy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_saxpy( const int N, const float alpha, const float *X,
                       const int incX, float *Y, const int incY)
{
   const char *saxpy_kernel = "\n" \
                  "__kernel void saxpy(__global float* X, __global float* Y, int N, int incX, int incY, float alpha) { \n" \
                  "  int id = get_global_id(0);\n" \
                  "  if (id < N) { \n" \
                  "     if (incX == 1 || (id % incX) == 0) { \n" \
                  "        Y[id] = X[id] * alpha + Y[id];\n" \
                  "     } \n" \
                  "  }\n" \
                  "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 0, 0, (void *)&alpha, saxpy_kernel, "saxpy");
} 
