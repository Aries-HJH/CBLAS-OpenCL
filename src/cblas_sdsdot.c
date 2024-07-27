/*
 * cblas_sdsdot.c
 *
 * The program is a C interface to sdsdot.
 * It calls the fortran wrapper before calling sdsdot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

float cblas_sdsdot( const int N, const float alpha, const float *X,
                      const int incX, const float *Y, const int incY)
{
   const char * sdsdot_kernel = "\n" \
         "__kernel void sdsdot(__global float* X, __global float* Y, __global float *res, int N, int incX, int incY) { \n" \
         "  int id = get_local_id(0); \n" \
         "  int gid = get_global_id(0); \n" \
         "  int local_size = get_local_size(0); \n" \
         "  __local float shader_Y[64]; \n" \
         "  if (gid < N) { \n" \
         "     if (incX == 1 || (gid % incX) == 0) { \n" \
         "        Y[gid] = X[gid] * Y[gid];\n" \
         "        shader_Y[id] = Y[gid];\n" \
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
   float res = 0;
   cblas_run_2vec_r(N, (void *)X, incX, (void *)Y, incY, (void *)&res, 0, 0, 0, NULL, sdsdot_kernel, "sdsdot");
   return res * alpha;
}   
