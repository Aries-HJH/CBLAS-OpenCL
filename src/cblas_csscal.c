/*
 * cblas_csscal.c
 *
 * The program is a C interface to cscal.f.
 *
 * Written by JiaHui Hou.  28/8/2024
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

void cblas_csscal( const int N, const float alpha, void *X, 
                       const int incX)
{
   const char *csscal_kernel = "\n" \
               "__kernel void csscal(__global float* X, int N, int incX, float alpha1, float alpha2) { \n" \
               "  int id = get_global_id(0);\n" \
               "  if (id < N) { \n" \
               "     if (incX == 1 || (id % incX) == 0) { \n" \
               "        X[id * 2] = X[id * 2] * alpha1;\n" \
               "        X[id * 2 + 1] = X[id * 2 + 1] * alpha2;\n" \
               "     } \n" \
               "  }\n" \
               "}";
   cblas_run_1vec(N, (void *)X, incX, 0, 1, &alpha, csscal_kernel, "csscal");
}
