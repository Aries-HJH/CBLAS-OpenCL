/*
 * cblas_zaxpy.c
 *
 * The program is a C interface to zaxpy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_zaxpy( const int N, const void *alpha, const void *X,
                       const int incX, void *Y, const int incY)
{
   const char *zaxpy_kernel = "\n" \
                  "#ifdef cl_khr_fp64 \n" \
                  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
                  "#elif defined(cl_amd_fp64) \n" \
                  "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
                  "#else \n" \
                  "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
                  "#endif \n" \
                  "__kernel void zaxpy(__global double* X, __global double* Y, int N, int incX, int incY, double alpha1, double alpha2) { \n" \
                  "  int id = get_global_id(0);\n" \
                  "  if (id < N) { \n" \
                  "     if (incX == 1 || (id % incX) == 0) { \n" \
                  "        Y[id * 2] = X[id * 2] * alpha1 + Y[id * 2];\n" \
                  "        Y[id * 2 + 1] = X[id * 2 + 1] * alpha2 + Y[id * 2 + 1];\n" \
                  "     } \n" \
                  "  }\n" \
                  "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 1, 1, alpha, zaxpy_kernel, "zaxpy");
} 
