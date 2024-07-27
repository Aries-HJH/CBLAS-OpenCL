/*
 * cblas_ddot.c
 *
 * The program is a C interface to ddot.
 * It calls the fortran wrapper before calling ddot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

double cblas_ddot( const int N, const double *X,
                      const int incX, const double *Y, const int incY)
{
   const char * ddot_kernel = "\n" \
         "#ifdef cl_khr_fp64 \n" \
         "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
         "#elif defined(cl_amd_fp64) \n" \
         "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
         "#else \n" \
         "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
         "#endif \n" \
         "__kernel void ddot(__global double* X, __global double* Y, __global double *res, int N, int incX, int incY) { \n" \
         "  int id = get_local_id(0); \n" \
         "  int gid = get_global_id(0); \n" \
         "  int local_size = get_local_size(0); \n" \
         "  __local double shader_Y[64]; \n" \
         "  if (gid < N) { \n" \
         "     if (incX == 1 || (gid % incX) == 0) { \n" \
         "        shader_Y[id] = (double)X[gid] * (double)Y[gid];\n" \
         "     } \n" \
         "  } \n" \
         "  barrier(CLK_LOCAL_MEM_FENCE); \n" \
         "  for (int i = local_size / 2; i > 0; i >>= 1) { \n" \
         "     if (id < i) { \n" \
         "        shader_Y[id] += shader_Y[id + i]; \n" \
         "     } \n" \
         "     barrier(CLK_LOCAL_MEM_FENCE); \n" \
         "  } \n" \
         "  if (id == 0) { \n" \
         "     res[0] += shader_Y[0]; \n" \
         "  } \n" \
         "}";
   double res = 0;
   cblas_run_2vec_r(N, (void *)X, incX, (void *)Y, incY, (void *)&res, 1, 1, 0, NULL, ddot_kernel, "ddot");
   return res;
}   
