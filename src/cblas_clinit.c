#include "cblas_clinit.h"
#include <stdio.h>
struct OpenCLInfo ocl_info;

/*
* name: cblas_clinit
* description: init OpenCL
*/
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

static void check_error(const int incX, const int incY, const int N)
{
    if (incX != incY || incX == 0 || incY == 0 || N < incX) {
        printf("ERROR: incX or incY set an error value.\n");
        return;
    }
}

static void create_program(const char *kernelSource, const char *name)
{
    ocl_info.cl_prog = clCreateProgramWithSource(ocl_info.cl_ctx, 1, (const char**)&kernelSource, NULL, NULL);
    ocl_info.cl_status = clBuildProgram(ocl_info.cl_prog, 1, &ocl_info.cl_device, NULL, NULL, NULL);
    if (ocl_info.cl_status != CL_SUCCESS) {
        size_t info_size;
        clGetProgramBuildInfo(ocl_info.cl_prog, ocl_info.cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &info_size);
        char * log = (char *)malloc(sizeof(char) * info_size);
        clGetProgramBuildInfo(ocl_info.cl_prog, ocl_info.cl_device, CL_PROGRAM_BUILD_LOG, info_size, (void *)log, NULL);
        printf("%s\n", log);
        free(log);
    }
    ocl_info.cl_kern = clCreateKernel(ocl_info.cl_prog, name, NULL);
}

/*
* name: cblas_run_2vec
* description: Two vector operation function
* TODO: 如果用户给入数据 double类型的X Y , float类型的alpha。这种情况需要处理
*/
void cblas_run_2vec(const int N, void* X, const int incX, void* Y, const int incY,
                    cl_bool is_double, cl_bool is_complex, const void *alpha, const char *kernelSource, const char *name)
{
    check_error(incX, incY, N);
    size_t mem_size;
    if (is_double) {
        mem_size = N * sizeof(double);
    } else {
        mem_size = N * sizeof(float);
    }
    if (is_complex)
        mem_size *= 2;
    cl_mem cl_x, cl_y;
    cl_x = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, mem_size, (void *)X, &ocl_info.cl_status);
    if (ocl_info.cl_status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL memory.\n");
        return;
    }
    cl_y = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, mem_size, (void *)Y, &ocl_info.cl_status);
    if (ocl_info.cl_status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL memory.\n");
        return;
    }
    create_program(kernelSource, name);
    clSetKernelArg(ocl_info.cl_kern, 0, sizeof(cl_mem), &cl_x);
    clSetKernelArg(ocl_info.cl_kern, 1, sizeof(cl_mem), &cl_y);
    clSetKernelArg(ocl_info.cl_kern, 2, sizeof(int), &N);
    clSetKernelArg(ocl_info.cl_kern, 3, sizeof(int), &incX);
    clSetKernelArg(ocl_info.cl_kern, 4, sizeof(int), &incY);
    if (alpha != NULL) {
        size_t bytes = is_double ? sizeof(double) : sizeof(float);
        clSetKernelArg(ocl_info.cl_kern, 5, bytes, alpha);
        if (is_complex) {
            clSetKernelArg(ocl_info.cl_kern, 6, bytes, alpha + bytes);
        }
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((N / local_work_size) + 1) * local_work_size;
    clEnqueueNDRangeKernel(ocl_info.cl_cq, ocl_info.cl_kern, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clFinish(ocl_info.cl_cq);
}

/* Two vector operation function and return value*/
void cblas_run_2vec_r(const int N, void* X, const int incX, void* Y, const int incY, void *res, cl_bool is_res_double,
                    cl_bool is_double, cl_bool is_complex, const void *alpha, const char *kernelSource, const char *name)
{
    check_error(incX, incY, N);
    size_t mem_size;
    if (is_double) {
        mem_size = N * sizeof(double);
    } else {
        mem_size = N * sizeof(float);
    }
    if (is_complex)
        mem_size *= 2;
    cl_mem cl_x, cl_y;
    cl_x = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, mem_size, X, &ocl_info.cl_status);
    if (ocl_info.cl_status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL memory.\n");
        return;
    }
    cl_y = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, mem_size, Y, &ocl_info.cl_status);
    if (ocl_info.cl_status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL memory.\n");
        return;
    }
    cl_mem cl_res;
    size_t bytes = is_res_double ? sizeof(double) : sizeof(float);
    if (is_complex)
        bytes *= 2;
    cl_res = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes, res, &ocl_info.cl_status);
    if (ocl_info.cl_status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL memory.\n");
        return;
    }
    create_program(kernelSource, name);
    clSetKernelArg(ocl_info.cl_kern, 0, sizeof(cl_mem), &cl_x);
    clSetKernelArg(ocl_info.cl_kern, 1, sizeof(cl_mem), &cl_y);
    clSetKernelArg(ocl_info.cl_kern, 2, sizeof(cl_mem), &cl_res);
    clSetKernelArg(ocl_info.cl_kern, 3, sizeof(int), &N);
    clSetKernelArg(ocl_info.cl_kern, 4, sizeof(int), &incX);
    clSetKernelArg(ocl_info.cl_kern, 5, sizeof(int), &incY);
    size_t local_work_size = 64;
    size_t global_work_size = ((N / local_work_size) + 1) * local_work_size;
    clEnqueueNDRangeKernel(ocl_info.cl_cq, ocl_info.cl_kern, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clFinish(ocl_info.cl_cq);
}

void cblas_run_1vec(const int N, void* X, const int incX, cl_bool is_double, cl_bool is_complex, 
                    const void *alpha, const char *kernelSource, const char *name)
{
    if (incX == 0 || incX > N) {
        printf("ERROR: Invalid value about incX.\n");
        return;
    }
    size_t mem_size;
    if (is_double) {
        mem_size = N * sizeof(double);
    } else {
        mem_size = N * sizeof(float);
    }
    if (is_complex)
        mem_size *= 2;
    cl_mem cl_x;
    cl_x = clCreateBuffer(ocl_info.cl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, mem_size, X, &ocl_info.cl_status);
    if (ocl_info.cl_status != CL_SUCCESS) {
        printf("ERROR: failed to create OpenCL memory.\n");
        return;
    }
    create_program(kernelSource, name);
    clSetKernelArg(ocl_info.cl_kern, 0, sizeof(cl_mem), &cl_x);
    clSetKernelArg(ocl_info.cl_kern, 1, sizeof(int), &N);
    clSetKernelArg(ocl_info.cl_kern, 2, sizeof(int), &incX);
    if (alpha != NULL) {
        size_t bytes = is_double ? sizeof(double) : sizeof(float);
        clSetKernelArg(ocl_info.cl_kern, 3, bytes, alpha);
        if (is_complex) {
            clSetKernelArg(ocl_info.cl_kern, 4, bytes, alpha + bytes);
        }
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((N / local_work_size) + 1) * local_work_size;
    clEnqueueNDRangeKernel(ocl_info.cl_cq, ocl_info.cl_kern, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clFinish(ocl_info.cl_cq);
}