#ifndef CBLAS_CLINIT_H
#define CBLAS_CLINIT_H
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

struct OpenCLInfo {
    cl_platform_id cl_platform;
    cl_device_id cl_device;
    cl_context cl_ctx;
    cl_command_queue cl_cq;
    cl_int cl_status;
    cl_program cl_prog;
    cl_kernel cl_kern;
};

void cblas_clinit();

#endif