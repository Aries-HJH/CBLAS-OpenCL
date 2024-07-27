/*
 * cblas_scopy.c
 *
 * The program is a C interface to scopy.
 *
 * Written by Xi'an China.  27/7/2024
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_scopy( const int N, const float *X,
                      const int incX, float *Y, const int incY)
{
   const char * scopy_kernel = "\n" \
         "__kernel void scopy(__global float* X, __global float* Y, int N, int incX, int incY) { \n" \
         "  int id = get_global_id(0);\n" \
         "  if (id < N) { \n" \
         "     if (incX == 1 || (id % incX) == 0) { \n" \
         "        Y[id] = X[id];\n" \
         "     } \n" \
         "  }\n" \
         "}";
   cblas_run_2vec(N, (void *)X, incX, (void *)Y, incY, 0, 0, NULL, scopy_kernel, "scopy");
}
