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

void cblas_run_2vec(const int N, void* X, const int incX, void* Y, const int incY,
                    cl_bool is_double, cl_bool is_complex, const void *alpha, const char *kernelSource, const char *name);

void cblas_run_2vec_r(const int N, void* X, const int incX, void* Y, const int incY, void *res, cl_bool is_res_double,
                    cl_bool is_double, cl_bool is_complex, const void *alpha, const char *kernelSource, const char *name);

void cblas_run_1vec(const int N, void* X, const int incX, cl_bool is_double, cl_bool is_complex, 
                    const void *alpha, const char *kernelSource, const char *name);

#endif