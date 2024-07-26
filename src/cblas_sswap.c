/*
 * cblas_sswap.c
 *
 * The program is a C interface to sswap.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_clinit.h"
#include <stdio.h>

extern struct OpenCLInfo ocl_info;

// TODO: 需要实现矩阵形式的swap
const char * kernel_source = "\n" \
                     "__kernel void sswap(__global float* X, __global float* Y, int N, int incX, int incY) { \n" \
                     "  int id = get_global_id(0);\n" \
                     "  float temp;\n" \
                     "  if (id < N) { \n" \
                     "     temp = X[id];\n" \
                     "     X[id] = Y[id];\n" \
                     "     Y[id] = temp;\n" \
                     "  }\n" \
                     "}";

void cblas_sswap( const int N, float *X, const int incX, float *Y,
                       const int incY)
{
   cl_int status;
   cl_mem cl_x, cl_y;
   cl_x = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), (void *)X, &status);
   if (status != CL_SUCCESS) {
      printf("ERROR: failed to create OpenCL memory.\n");
      return;
   }
   cl_y = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), (void *)Y, &status);
   if (status != CL_SUCCESS) {
      printf("ERROR: failed to create OpenCL memory.\n");
      return;
   }
   cl_program program;
   program = clCreateProgramWithSource(ocl_info.cl_ctx, 1, (const char**)&kernel_source, NULL, NULL);
   status = clBuildProgram(program, 1, &ocl_info.cl_device, NULL, NULL, NULL);
   if (status != CL_SUCCESS) {
      size_t info_size;
      clGetProgramBuildInfo(program, ocl_info.cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &info_size);
      char * log = (char *)malloc(sizeof(char) * info_size);
      clGetProgramBuildInfo(program, ocl_info.cl_device, CL_PROGRAM_BUILD_LOG, info_size, (void *)log, NULL);
      printf("%s\n", log);
      free(log);
   }
   cl_kernel kernel;
   kernel = clCreateKernel(program, "sswap", NULL);
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_x);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_y);
   clSetKernelArg(kernel, 2, sizeof(int), &N);
   clSetKernelArg(kernel, 3, sizeof(int), &incX);
   clSetKernelArg(kernel, 4, sizeof(int), &incY);
   size_t local_work_size = 256;
   size_t global_work_size = ((N / 256) + 1) * 256;
   clEnqueueNDRangeKernel(ocl_info.cl_cq, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
   clFinish(ocl_info.cl_cq);

}

