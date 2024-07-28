/*
 * cblas_cscal.c
 *
 * The program is a C interface to cscal.f.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_cscal( const int N, const void *alpha, void *X, 
                       const int incX)
{
   const char *cscal_kernel = "\n" \
               "__kernel void cscal(__global float* X, int N, int incX, float alpha1, float alpha2) { \n" \
               "  int id = get_global_id(0);\n" \
               "  if (id < N) { \n" \
               "     if (incX == 1 || (id % incX) == 0) { \n" \
               "        X[id * 2] = X[id * 2] * alpha1;\n" \
               "        X[id * 2 + 1] = X[id * 2 + 1] * alpha2;\n" \
               "     } \n" \
               "  }\n" \
               "}";
   cblas_run_1vec(N, (void *)X, incX, 0, 1, alpha, cscal_kernel, "cscal");
}
