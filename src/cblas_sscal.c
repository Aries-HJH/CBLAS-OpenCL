/*
 * cblas_sscal.c
 *
 * The program is a C interface to sscal.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_sscal( const int N, const float alpha, float *X, 
                       const int incX)
{
   const char *sscal_kernel = "\n" \
               "__kernel void sscal(__global float* X, int N, int incX, float alpha) { \n" \
               "  int id = get_global_id(0);\n" \
               "  if (id < N) { \n" \
               "     if (incX == 1 || (id % incX) == 0) { \n" \
               "        X[id] = X[id] * alpha;\n" \
               "     } \n" \
               "  }\n" \
               "}";
   cblas_run_1vec(N, (void *)X, incX, 0, 0, (void*)&alpha, sscal_kernel, "sscal");
}
