//OPENCL DONE
int gpu_index;
#ifdef OPENCL

#define PLATFORM_ID 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef CLBLAS
#include <clBLAS.h>
#endif

#include "openclutils.h"
#include "utils.h"
#include "activations.h"

cl_info cl = {0};

void cl_check_error(cl_info info)
{
    clFinish(cl.queue);
    if (info.error != CL_SUCCESS)
    {
        printf("\n Error number %d \n", info.error);
        abort();
        exit(1);
    }
}

#define MAX_DEVICES 10

void cl_set_device(cl_info *info, int index)
{
    if (!info->initialized || (info->current_device_id!=index)) {
        cl_uint num_devices;
        cl_device_id devices[MAX_DEVICES];
        info->error=clGetDeviceIDs(info->platform, CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
        cl_check_error(*info);

        index = index%num_devices;
        info->device = devices[index];
        cl_check_error(*info);

        cl_context_properties properties[]=
        {
            CL_CONTEXT_PLATFORM, (cl_context_properties)info->platform, 0
        };

        // Note that nVidia's OpenCL requires the platform property
        info->context=clCreateContext(properties, 1, &info->device, 0, 0, &info->error);
        cl_check_error(*info);

        info->queue = clCreateCommandQueue(info->context, info->device, 0, &info->error);
        cl_check_error(*info);
    }
}

cl_info cl_init(int index)
{
    cl_info info;
    info.initialized = 0;
    cl_platform_id *platform_id_array;
    if(index < 0) error("Won't initialize negative gpu id\n");
    cl_uint num_platforms, num_devices;
    // Fetch the Platform and Device IDs;
    cl_device_id devices[MAX_DEVICES];

    info.error=clGetPlatformIDs(5, &info.platform, &num_platforms);
    cl_check_error(info);

    platform_id_array = (cl_platform_id *)calloc(num_platforms, sizeof(cl_platform_id));

    info.error = clGetPlatformIDs(num_platforms, platform_id_array, NULL);

    info.platform = platform_id_array[PLATFORM_ID];     //select the platform
    cl_check_error(info);
    cl_set_device(&info, index);
    /*
    info.error=clGetDeviceIDs(info.platform, CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
    cl_check_error(info);

    index = index%num_devices;
    info.device = devices[index];
    cl_check_error(info);

    cl_context_properties properties[]=
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)info.platform, 0
    };

    // Note that nVidia's OpenCL requires the platform property
    info.context=clCreateContext(properties, 1, &info.device, 0, 0, &info.error);
    cl_check_error(info);

    info.queue = clCreateCommandQueue(info.context, info.device, 0, &info.error);
    cl_check_error(info);*/
#ifdef CLBLAS
    info.error = clblasSetup();
#endif
    cl_check_error(info);
    info.initialized = 1;

//#ifdef VERBOSE
    printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);
    char buffer[10240];
    clGetPlatformInfo(info.platform, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
    printf("  PROFILE = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VERSION, 10240, buffer, NULL);
    printf("  VERSION = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_NAME, 10240, buffer, NULL);
    printf("  NAME = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
    printf("  VENDOR = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
    printf("  EXTENSIONS = %s\n", buffer);
    cl_check_error(info);

    if(num_devices > MAX_DEVICES) num_devices = MAX_DEVICES;
    printf("=== %d OpenCL device(s) found on platform:\n", num_devices);
    int i;
    for (i=0; i<num_devices; i++)
    {
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        printf("  -- %d --\n", i);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_NAME = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VENDOR = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DRIVER_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_LOCAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  CL_KERNEL_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_MAX_MEM_ALLOC_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_MAX_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
        cl_uint items;
        clGetDeviceInfo( devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
                         &items, NULL);
        printf("  DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned int)items);
        size_t workitem_size[10];
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 10*sizeof(workitem_size), workitem_size, NULL);
        printf("  DEVICE_MAX_WORK_ITEM_SIZES = %u / %u / %u \n", (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);
        printf("%d devices, %d index\n", num_devices, index);

    }
//#endif
    return info;
}

cl_program cl_fprog(char *filename, char *options, cl_info info)
{
    size_t srcsize;
    char src[64*1024];
    memset(src, 0, 64*1024);
    FILE *fil=fopen(filename,"r");
    if(fil == 0) file_error(filename);
    srcsize=fread(src, sizeof src, 1, fil);
    fclose(fil);
    const char *srcptr[]= {src};
    // Submit the source code of the example kernel to OpenCL
    cl_program prog=clCreateProgramWithSource(info.context,1, srcptr, &srcsize, &info.error);
    cl_check_error(info);
    char build_c[1024*64];
    // and compile it (after this we could extract the compiled version)
    info.error=clBuildProgram(prog, 0, 0, options, 0, 0);
    if ( info.error != CL_SUCCESS )
    {
        fprintf(stderr, "Error Building Program: %d\n", info.error);
        clGetProgramBuildInfo( prog, info.device, CL_PROGRAM_BUILD_LOG, 1024*64, build_c, 0);
        fprintf(stderr, "Build Log for %s program:\n%s\n", filename, build_c);
    }
    cl_check_error(info);
    return prog;
}

void cl_setup(int index)
{
    if(!cl.initialized)
    {
        fprintf(stderr, "Initializing OpenCL\n");
        gpu_index = index;
        cl = cl_init(gpu_index);
    }
}

cl_kernel get_kernel(char *filename, char *kernelname, char *options)
{
    cl_program prog = cl_fprog(filename, options, cl);
    cl_kernel kernel=clCreateKernel(prog, kernelname, &cl.error);
    cl_check_error(cl);
    return kernel;
}
//cl_random added by S. Li
void cl_random(cl_mem_with_offset mem, int n)
{
    float *x = calloc(n, sizeof(float));
    int i;
    for (i = 0; i < n; ++i)
    {
        //x[i] = rand_uniform();
        x[i]=(float)rand()/RAND_MAX;
    }
    cl_write_array(mem, x, n);
    free(x);
}

void cl_read_array(cl_mem_with_offset mem, float *x, int n)
{
    if(gpu_index < 0) return;
    cl.error = clEnqueueReadBuffer(cl.queue, mem.memory, CL_TRUE,(size_t)(sizeof(float)*mem.offset), (size_t)(sizeof(float)*n),x,0,0,0);
    cl_check_error(cl);
}

float cl_sum_array(cl_mem_with_offset mem, size_t n)
{

    float *x = calloc(n, sizeof(float));
    cl_read_array(mem, x, n);
    float sum = sum_array(x, n);
    free(x);
    return sum;
}

float cl_mag_array(cl_mem_with_offset mem, size_t n)
{

    float *x = calloc(n, sizeof(float));
    cl_read_array(mem, x, n);
    float m = mag_array(x, n);
    free(x);
    return m;
}

void cl_write_array(cl_mem_with_offset mem, float *x, int n)
{
    if(gpu_index < 0) return;
    cl.error = clEnqueueWriteBuffer(cl.queue, mem.memory, CL_TRUE, sizeof(float)*mem.offset,sizeof(float)*n,x,0,0,0);
    cl_check_error(cl);
}

void cl_copy_array(cl_mem_with_offset src, cl_mem_with_offset dst, int n)
{
    cl.error = clEnqueueCopyBuffer(cl.queue, src.memory, dst.memory, sizeof(float)*src.offset, sizeof(float)*dst.offset, sizeof(float)*n,0,0,0);
    cl_check_error(cl);
}

cl_mem_with_offset cl_sub_array(cl_mem_with_offset src, int offset, int size)
{
    cl_buffer_region r;
    r.origin = offset*sizeof(float);
    r.size = size*sizeof(float);
    cl_mem_with_offset sub;
    sub.memory = clCreateSubBuffer(src.memory, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &cl.error);
    sub.offset = 0;
    if(size)
        sub.exist = 1;
    else
        sub.exist = 0;
    cl_check_error(cl);
    return sub;
}


cl_mem_with_offset cl_make_array(float *x, size_t n)
{
    cl_mem_with_offset mem;

    if(gpu_index < 0)
    {
        mem.exist = 0;
        return mem;
    }
    if(x)
    {
        mem.memory = clCreateBuffer(cl.context,
                                    CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                    sizeof(float)*n, x, &cl.error);
        mem.offset = 0;
        if (n)
            mem.exist = 1;
        else
            mem.exist = 0;
    }

    else
    {
        mem.memory = clCreateBuffer(cl.context,
                                    CL_MEM_READ_WRITE,
                                    sizeof(float)*((int)(n*1.1)), x, &cl.error);
        mem.offset = 0;
        if(n)
            mem.exist = 1;
        else
            mem.exist = 0;
    }

    cl_check_error(cl);
    return mem;
}

cl_mem_with_offset cl_make_int_array(int *x, int n)
{
    cl_mem_with_offset mem;
    if(gpu_index < 0)
    {
        mem.exist = 0;
        return mem;
    }

    mem.offset = 0;
    if(x)
    {
        mem.memory = clCreateBuffer(cl.context,
                                    CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                    sizeof(int)*n, x, &cl.error);
        mem.offset = 0;
        if (n)
            mem.exist = 1;
        else
            mem.exist = 0;
    }

    else
    {
        mem.memory = clCreateBuffer(cl.context,
                                    CL_MEM_READ_WRITE,
                                    sizeof(int)*((int)(n*1.1)), x, &cl.error);
        mem.offset = 0;
        if(n)
            mem.exist = 1;
        else
            mem.exist = 0;
    }
    cl_check_error(cl);
    return mem;
}

cl_int cl_free(cl_mem_with_offset memobj)
{
    memobj.offset = 0;
    memobj.exist = 0;
    return clReleaseMemObject(memobj.memory);
}

int cl_global_size(int x, int blocknum)
{
    return (int)((x-1)/blocknum+1)*blocknum;
}


#endif
