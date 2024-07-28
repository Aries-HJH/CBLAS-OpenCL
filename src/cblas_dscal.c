/*
 * cblas_dscal.c
 *
 * The program is a C interface to dscal.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_dscal( const int N, const double alpha, double *X, 
                       const int incX)
{
   const char *dscal_kernel = "\n" \
               "#ifdef cl_khr_fp64 \n" \
               "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
               "#elif defined(cl_amd_fp64) \n" \
               "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
               "#else \n" \
               "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
               "#endif \n" \
               "__kernel void dscal(__global double* X, int N, int incX, double alpha) { \n" \
               "  int id = get_global_id(0);\n" \
               "  if (id < N) { \n" \
               "     if (incX == 1 || (id % incX) == 0) { \n" \
               "        X[id] = X[id] * alpha;\n" \
               "     } \n" \
               "  }\n" \
               "}";
   cblas_run_1vec(N, (void *)X, incX, 1, 0, (void*)&alpha, dscal_kernel, "dscal");
}
