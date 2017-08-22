//OPENCL done
#ifndef OPENCLUTIL_H
#define OPENCLUTIL_H
extern int gpu_index;
#ifdef OPENCL
#define GPU

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdbool.h>

#define BLOCK 256
#define BLOCK1 16 //BLOCK size for original gemm
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

typedef struct {
    int initialized;
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    int current_device_id;
}cl_info;

typedef struct {
    cl_mem memory;
    int offset;
    bool exist;
}cl_mem_with_offset;

extern cl_info cl;

void cl_setup(int index);
void cl_check_error(cl_info info);
cl_kernel get_kernel(char *filename, char *kernelname, char *options);
void cl_random(cl_mem_with_offset mem, int n);
void cl_read_array(cl_mem_with_offset mem, float *x, int n);
void cl_write_array(cl_mem_with_offset mem, float *x, int n);
cl_mem_with_offset cl_make_array(float *x, size_t n);
cl_mem_with_offset cl_make_int_array(int *x, int n);
void cl_copy_array(cl_mem_with_offset src, cl_mem_with_offset dst, int n);
cl_mem_with_offset cl_sub_array(cl_mem_with_offset src, int offset, int size);
float cl_sum_array(cl_mem_with_offset mem, size_t n);
float cl_mag_array(cl_mem_with_offset mem, size_t n);
void cl_set_device(cl_info* info, int index);
cl_info cl_init(int index);
cl_int cl_free(cl_mem_with_offset memobj);

int cl_global_size(int x, int blocknum);

#endif
#endif
