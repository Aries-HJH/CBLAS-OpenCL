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

const char * kernel_source = "\n" \
                     "__kernel void sswap(__global float* X, __global float* Y, int N, int incX, int incY) { \n" \
                     "  int id = get_global_id(0);\n" \
                     "  float temp;\n" \
                     "  if (id < N) { \n" \
                     "     if (incX == 1 || (id % incX) == 0) { \n" \
                     "        temp = X[id];\n" \
                     "        X[id] = Y[id];\n" \
                     "        Y[id] = temp;\n" \
                     "     } \n" \
                     "  }\n" \
                     "}";

void cblas_sswap( const int N, float *X, const int incX, float *Y,
                       const int incY)
{
   if (incX != incY || incX == 0 || incY == 0) {
      printf("ERROR: incX or incY set an error value.\n");
      return;
   }
   cl_mem cl_x, cl_y;
   cl_x = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), (void *)X, &ocl_info.cl_status);
   if (ocl_info.cl_status != CL_SUCCESS) {
      printf("ERROR: failed to create OpenCL memory.\n");
      return;
   }
   cl_y = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), (void *)Y, &ocl_info.cl_status);
   if (ocl_info.cl_status != CL_SUCCESS) {
      printf("ERROR: failed to create OpenCL memory.\n");
      return;
   }
   
   ocl_info.cl_prog = clCreateProgramWithSource(ocl_info.cl_ctx, 1, (const char**)&kernel_source, NULL, NULL);
   ocl_info.cl_status = clBuildProgram(ocl_info.cl_prog, 1, &ocl_info.cl_device, NULL, NULL, NULL);
   if (ocl_info.cl_status != CL_SUCCESS) {
      size_t info_size;
      clGetProgramBuildInfo(ocl_info.cl_prog, ocl_info.cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &info_size);
      char * log = (char *)malloc(sizeof(char) * info_size);
      clGetProgramBuildInfo(ocl_info.cl_prog, ocl_info.cl_device, CL_PROGRAM_BUILD_LOG, info_size, (void *)log, NULL);
      printf("%s\n", log);
      free(log);
   }
   ocl_info.cl_kern = clCreateKernel(ocl_info.cl_prog, "sswap", NULL);
   clSetKernelArg(ocl_info.cl_kern, 0, sizeof(cl_mem), &cl_x);
   clSetKernelArg(ocl_info.cl_kern, 1, sizeof(cl_mem), &cl_y);
   clSetKernelArg(ocl_info.cl_kern, 2, sizeof(int), &N);
   clSetKernelArg(ocl_info.cl_kern, 3, sizeof(int), &incX);
   clSetKernelArg(ocl_info.cl_kern, 4, sizeof(int), &incY);
   size_t local_work_size = 64;
   size_t global_work_size = ((N / local_work_size) + 1) * local_work_size;
   clEnqueueNDRangeKernel(ocl_info.cl_cq, ocl_info.cl_kern, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
   clFinish(ocl_info.cl_cq);

}

