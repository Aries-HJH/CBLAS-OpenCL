#include "cblas_clinit.h"
#include <stdio.h>
struct OpenCLInfo ocl_info;

void cblas_clinit()
{
    cl_uint status;
    status = clGetPlatformIDs(1, &ocl_info.cl_platform, NULL);
    if (status != CL_SUCCESS) {
        printf("ERROR: failed to find any platform.\n");
        return;
    }
    // TODO: 在CPU和GPU类型的OpenCL自动检测
    cl_uint device_num;
    clGetDeviceIDs(ocl_info.cl_platform, CL_DEVICE_TYPE_CPU, 1, &ocl_info.cl_device, &device_num);
    if (device_num == 0) {
        printf("ERROR: failed to find any device.\n");
        return;
    }
    ocl_info.cl_ctx = clCreateContext(NULL, 1, &ocl_info.cl_device, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL context.\n");
        return;
    }
    ocl_info.cl_cq = clCreateCommandQueue(ocl_info.cl_ctx, ocl_info.cl_device, CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL CommandQueue.\n");
        return;
    }
}