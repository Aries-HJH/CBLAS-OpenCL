/*
 * cblas_zswap.c
 *
 * The program is a C interface to zswap.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_zswap( const int N, void  *X, const int incX, void  *Y,
                       const int incY)
{
   const char *zswap_kernel = "\n" \
                     "#ifdef cl_khr_fp64 \n" \
                     "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
                     "#elif defined(cl_amd_fp64) \n" \
                     "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
                     "#else \n" \
                     "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
                     "#endif \n" \
                     "__kernel void zswap(__global double* X, __global double* Y, int N, int incX, int incY) { \n" \
                     "  int id = get_global_id(0);\n" \
                     "  double temp;\n" \
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
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 1, 1, NULL, zswap_kernel, "zswap");
}
