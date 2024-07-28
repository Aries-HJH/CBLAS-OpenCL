/*
 * cblas_zscal.c
 *
 * The program is a C interface to zscal.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_zscal( const int N, const void *alpha, void *X, 
                       const int incX)
{
   const char *zscal_kernel = "\n" \
               "#ifdef cl_khr_fp64 \n" \
               "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
               "#elif defined(cl_amd_fp64) \n" \
               "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
               "#else \n" \
               "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
               "#endif \n" \
               "__kernel void zscal(__global double* X, int N, int incX, double alpha1, double alpha2) { \n" \
               "  int id = get_global_id(0);\n" \
               "  if (id < N) { \n" \
               "     if (incX == 1 || (id % incX) == 0) { \n" \
               "        X[id * 2] = X[id * 2] * alpha1;\n" \
               "        X[id * 2 + 1] = X[id * 2 + 1] * alpha2;\n" \
               "     } \n" \
               "  }\n" \
               "}";
   cblas_run_1vec(N, (void *)X, incX, 1, 1, alpha, zscal_kernel, "zscal");
}
