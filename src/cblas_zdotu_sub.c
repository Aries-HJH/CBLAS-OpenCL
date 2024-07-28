/*
 * cblas_zdotu_sub.c
 *
 * The program is a C interface to zdotu.
 * It calls the fortran wrapper before calling zdotu.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_zdotu_sub( const int N, const void *X, const int incX,
                      const void *Y, const int incY, void *dotu)
{
   const char * zdotu_sub_kernel = "\n" \
         "#ifdef cl_khr_fp64 \n" \
         "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
         "#elif defined(cl_amd_fp64) \n" \
         "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n " \
         "#else \n" \
         "#error \"Double precision floating point not supported by OpenCL implementation.\" \n " \
         "#endif \n" \
         "__kernel void zdotu_sub(__global double* X, __global double* Y, __global double *dotu, int N, int incX, int incY) { \n" \
         "  int id = get_local_id(0); \n" \
         "  int gid = get_global_id(0); \n" \
         "  int local_size = get_local_size(0); \n" \
         "  __local double shader_Y[128]; \n" \
         "  if (gid < N) { \n" \
         "     if (incX == 1 || (gid % incX) == 0) { \n" \
         "        shader_Y[id * 2] = X[gid * 2] * Y[gid * 2] - X[gid * 2 + 1] * Y[gid * 2 + 1];\n" \
         "        shader_Y[id * 2 + 1] = X[gid * 2 + 1] * Y[gid * 2] + X[gid * 2] * Y[gid * 2 + 1];\n" \
         "     } \n" \
         "  } \n" \
         "  barrier(CLK_LOCAL_MEM_FENCE); \n" \
         "  for (int i = local_size / 2; i > 0; i >>= 1) { \n" \
         "     if (id < i) { \n" \
         "        shader_Y[id * 2] += shader_Y[(id + i) * 2 + 1]; \n" \
         "        shader_Y[id * 2 + 1] += shader_Y[(id + i) * 2 + 1]; \n" \

         "     } \n" \
         "     barrier(CLK_LOCAL_MEM_FENCE); \n" \
         "  } \n" \
         "  if (id == 0) { \n" \
         "     dotu[0] += shader_Y[0]; \n" \
         "     dotu[1] += shader_Y[1]; \n" \
         "  } \n" \
         "}";
   cblas_run_2vec_r(N, (void *)X, incX, (void *)Y, incY, dotu, 1, 1, 1, NULL, zdotu_sub_kernel, "zdotu_sub");
}
