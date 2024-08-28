/*
 * cblas_srotg.c
 *
 * The program is a C interface to srotg.
 *
 * Written by JiaHui Hou.  24/8/2024
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"

#include <math.h>

// 制造一个吉文斯旋转矩阵，然后旋转
/*
a: 返回值 r 表示向量长度
b: 返回值 z 表示zero
c: 返回 cos tha
s: 返回 sin tha
向量： [a b]T
*/
void cblas_srotg(  float *a, float *b, float *c, float *s)
{
    const char *sscal_kernel = "\n" \
                "__kernel void sscal(__global float* a, __global float* b, __global float *c, __global float *s) { \n" \
                "  int id = get_global_id(0);\n" \
                "  r = pow(a * a + b * b, 0.5);" \
                "  " \
                "}";
    
}