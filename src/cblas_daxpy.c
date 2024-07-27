/*
 * cblas_daxpy.c
 *
 * The program is a C interface to daxpy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_daxpy( const int N, const double alpha, const double *X,
                       const int incX, double *Y, const int incY)
{
   const char *daxpy_kernel = "\n" \
                  "#ifdef cl_khr_fp64 \n" \
                  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
                  "#elif defined(cl_amd_fp64) \n" \
                  "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
                  "#else \n" \
                  "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
                  "#endif \n" \
                  "__kernel void daxpy(__global double* X, __global double* Y, int N, int incX, int incY, double alpha) { \n" \
                  "  int id = get_global_id(0);\n" \
                  "  if (id < N) { \n" \
                  "     if (incX == 1 || (id % incX) == 0) { \n" \
                  "        Y[id] = X[id] * alpha + Y[id];\n" \
                  "     } \n" \
                  "  }\n" \
                  "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 1, 0, (void *)&alpha, daxpy_kernel, "daxpy");
} 
