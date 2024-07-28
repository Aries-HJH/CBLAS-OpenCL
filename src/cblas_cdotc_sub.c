/*
 * cblas_cdotc_sub.c
 *
 * The program is a C interface to cdotc.
 * It calls the fortran wrapper before calling cdotc.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_cdotc_sub( const int N, const void *X, const int incX,
                    const void *Y, const int incY,void *dotc)
{
   const char * cdotc_sub_kernel = "\n" \
         "__kernel void cdotc_sub(__global float* X, __global float* Y, __global float *dotu, int N, int incX, int incY) { \n" \
         "  int id = get_local_id(0); \n" \
         "  int gid = get_global_id(0); \n" \
         "  int local_size = get_local_size(0); \n" \
         "  __local float shader_Y[128]; \n" \
         "  if (gid < N) { \n" \
         "     if (incX == 1 || (gid % incX) == 0) { \n" \
         "        shader_Y[id * 2] = X[gid * 2] * Y[gid * 2] + X[gid * 2 + 1] * Y[gid * 2 + 1];\n" \
         "        shader_Y[id * 2 + 1] = X[gid * 2 + 1] * Y[gid * 2] - X[gid * 2] * Y[gid * 2 + 1];\n" \
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
   cblas_run_2vec_r(N, (void *)X, incX, (void *)Y, incY, dotc, 0, 0, 1, NULL, cdotc_sub_kernel, "cdotc_sub");
}
