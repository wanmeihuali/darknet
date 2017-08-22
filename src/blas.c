#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef OPENCL
#include "openclutils.h"
#include <assert.h>

#endif

void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b)
    {
        for(k = 0; k < c; ++k)
        {
            for(j = 0; j < h; ++j)
            {
                for(i = 0; i < w; ++i)
                {
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b)
    {
        for(c = 0; c < layers; ++c)
        {
            for(i = 0; i < size; ++i)
            {
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b)
    {
        for(k = 0; k < minc; ++k)
        {
            for(j = 0; j < minh; ++j)
            {
                for(i = 0; i < minw; ++i)
                {
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i)
    {
        mean[i] = 0;
        for(j = 0; j < batch; ++j)
        {
            for(k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i)
    {
        variance[i] = 0;
        for(j = 0; j < batch; ++j)
        {
            for(k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b)
    {
        for(f = 0; f < filters; ++f)
        {
            for(i = 0; i < spatial; ++i)
            {
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j)
    {
        for(i = 0; i < NX; ++i)
        {
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i)
        {
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j)
    {
        for(i = 0; i < NX; ++i)
        {
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i)
        {
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1)
        {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else
        {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i)
    {
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i)
    {
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i)
    {
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b)
    {
        for(g = 0; g < groups; ++g)
        {
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

#ifdef OPENCL
cl_kernel get_scale_bias_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "scale_bias_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void scale_bias_gpu(cl_mem_with_offset output, cl_mem_with_offset biases, int batch, int n, int size)
{


    cl_kernel kernel = get_scale_bias_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.memory), (void*) &(output.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.offset), (void*) &(output.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases.memory), (void*) &(biases.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases.offset), (void*) &(biases.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(size,BLOCK), n, batch};
    const size_t localws[] = {BLOCK, 1, 1};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 3, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_backward_scale_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "backward_scale_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void backward_scale_gpu(cl_mem_with_offset x_norm, cl_mem_with_offset delta, int batch, int n, int size, cl_mem_with_offset scale_updates)
{



    cl_kernel kernel = get_backward_scale_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x_norm.memory), (void*) &(x_norm.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x_norm.offset), (void*) &(x_norm.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(scale_updates.memory), (void*) &(scale_updates.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(scale_updates.offset), (void*) &(scale_updates.offset));
    cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_add_bias_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "add_bias_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void add_bias_gpu(cl_mem_with_offset output, cl_mem_with_offset biases, int batch, int n, int size)
{


    cl_kernel kernel = get_add_bias_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.memory), (void*) &(output.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.offset), (void*) &(output.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases.memory), (void*) &(biases.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases.offset), (void*) &(biases.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(size,BLOCK), n, batch};
    const size_t localws[] = {BLOCK, 1, 1};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 3, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_backward_bias_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "backward_bias_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void backward_bias_gpu(cl_mem_with_offset bias_updates, cl_mem_with_offset delta, int batch, int n, int size)
{


    cl_kernel kernel = get_backward_bias_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(bias_updates.memory), (void*) &(bias_updates.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(bias_updates.offset), (void*) &(bias_updates.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_adam_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "adam_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void adam_gpu(int n, cl_mem_with_offset x, cl_mem_with_offset m, cl_mem_with_offset v, float B1, float B2, float rate, float eps, int t)
{



    cl_kernel kernel = get_adam_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(m.memory), (void*) &(m.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(m.offset), (void*) &(m.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(v.memory), (void*) &(v.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(v.offset), (void*) &(v.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(B1), (void*) &B1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(B2), (void*) &B2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(rate), (void*) &rate);
    cl.error = clSetKernelArg(kernel, i++, sizeof(eps), (void*) &eps);
    cl.error = clSetKernelArg(kernel, i++, sizeof(t), (void*) &t);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(n,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_normalize_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "normalize_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void normalize_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, cl_mem_with_offset variance, int batch, int filters, int spatial)
{



    int N = batch*filters*spatial;
    cl_kernel kernel = get_normalize_kernel();
    cl_command_queue queue = cl.queue;


    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_normalize_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "normalize_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void normalize_delta_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, cl_mem_with_offset variance, cl_mem_with_offset mean_delta, cl_mem_with_offset variance_delta, int batch, int filters, int spatial, cl_mem_with_offset delta)
{







    cl_kernel kernel = get_normalize_delta_kernel();
    cl_command_queue queue = cl.queue;
    size_t N = batch*filters*spatial;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta.memory), (void*) &(mean_delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta.offset), (void*) &(mean_delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance_delta.memory), (void*) &(variance_delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance_delta.offset), (void*) &(variance_delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_fast_mean_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "fast_mean_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_mean_delta_gpu(cl_mem_with_offset delta, cl_mem_with_offset variance, int batch, int filters, int spatial, cl_mem_with_offset mean_delta)
{




    cl_kernel kernel = get_fast_mean_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta.memory), (void*) &(mean_delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta.offset), (void*) &(mean_delta.offset));

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_fast_variance_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "fast_variance_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_variance_delta_gpu(cl_mem_with_offset x, cl_mem_with_offset delta, cl_mem_with_offset mean, cl_mem_with_offset variance, int batch, int filters, int spatial, cl_mem_with_offset variance_delta)
{






    cl_kernel kernel = get_fast_variance_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance_delta.memory), (void*) &(variance_delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance_delta.offset), (void*) &(variance_delta.offset));

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_mean_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "mean_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mean_delta_gpu(cl_mem_with_offset delta, cl_mem_with_offset variance, int batch, int filters, int spatial, cl_mem_with_offset mean_delta)
{




    cl_kernel kernel = get_mean_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta.memory), (void*) &(mean_delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta.offset), (void*) &(mean_delta.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(filters,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_mean_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "mean_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mean_gpu(cl_mem_with_offset x, int batch, int filters, int spatial, cl_mem_with_offset mean)
{



    cl_kernel kernel = get_mean_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(filters,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_variance_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "variance_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void variance_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, int batch, int filters, int spatial, cl_mem_with_offset variance)
{



    cl_kernel kernel = get_variance_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(filters,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_reorg_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "reorg_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void reorg_gpu(cl_mem_with_offset x, int w, int h, int c, int batch, int stride, int forward, cl_mem_with_offset out)
{



    cl_kernel kernel = get_reorg_kernel();
    cl_command_queue queue = cl.queue;
    int size = w*h*c*batch;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(w), (void*) &w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(h), (void*) &h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c), (void*) &c);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(forward), (void*) &forward);
    cl.error = clSetKernelArg(kernel, i++, sizeof(out.memory), (void*) &(out.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(out.offset), (void*) &(out.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(size,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_axpy_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "axpy_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void axpy_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX, cl_mem_with_offset Y, int INCY)
{

    cl_kernel kernel = get_axpy_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.memory), (void*) &(Y.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.offset), (void*) &(Y.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}






cl_kernel get_pow_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "pow_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void pow_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX, cl_mem_with_offset Y, int INCY)
{



    cl_kernel kernel = get_pow_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.memory), (void*) &(Y.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.offset), (void*) &(Y.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_const_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "const_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void const_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX)
{


    cl_kernel kernel = get_const_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_constrain_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "constrain_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void constrain_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX)
{


    cl_kernel kernel = get_constrain_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_supp_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "supp_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void supp_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX)
{


    cl_kernel kernel = get_supp_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_add_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "add_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void add_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX)
{


    cl_kernel kernel = get_add_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_scal_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "scal_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void scal_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX)
{


    cl_kernel kernel = get_scal_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_fill_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "fill_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fill_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX)
{


    cl_kernel kernel = get_fill_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    size_t a;
    clGetMemObjectInfo(X.memory, CL_MEM_SIZE, sizeof(size_t), &a, NULL);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);

    cl_check_error(cl);
}

cl_kernel get_mask_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "mask_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mask_gpu(int N, cl_mem_with_offset X, float mask_num, cl_mem_with_offset mask)
{


    cl_kernel kernel = get_mask_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask_num), (void*) &mask_num);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask.memory), (void*) &(mask.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask.offset), (void*) &(mask.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_copy_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "copy_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void copy_gpu(int N, cl_mem_with_offset X, /*int OFFX, */int INCX, cl_mem_with_offset Y, /*int OFFY,*/ int INCY)
{


    cl_kernel kernel = get_copy_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.memory), (void*) &(Y.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.offset), (void*) &(Y.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}






cl_kernel get_mul_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "mul_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mul_gpu(int N, cl_mem_with_offset X, int INCX, cl_mem_with_offset Y, int INCY)
{


    cl_kernel kernel = get_mul_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.memory), (void*) &(Y.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y.offset), (void*) &(Y.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_fast_mean_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "fast_mean_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_mean_gpu(cl_mem_with_offset x, int batch, int filters, int spatial, cl_mem_with_offset mean)
{


    cl_kernel kernel = get_fast_mean_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_fast_variance_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "fast_variance_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_variance_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, int batch, int filters, int spatial, cl_mem_with_offset variance)
{



    cl_kernel kernel = get_fast_variance_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.memory), (void*) &(mean.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean.offset), (void*) &(mean.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.memory), (void*) &(variance.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance.offset), (void*) &(variance.offset));

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_flatten_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "flatten_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void flatten_gpu(cl_mem_with_offset x, int spatial, int layers, int batch, int forward, cl_mem_with_offset out)
{



    cl_kernel kernel = get_flatten_kernel();
    cl_command_queue queue = cl.queue;
    int size = spatial*batch*layers;
    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.memory), (void*) &(x.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(x.offset), (void*) &(x.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layers), (void*) &layers);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(forward), (void*) &forward);
    cl.error = clSetKernelArg(kernel, i++, sizeof(out.memory), (void*) &(out.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(out.offset), (void*) &(out.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(size,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_shortcut_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "shortcut_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem_with_offset add, int w2, int h2, int c2, cl_mem_with_offset out)
{



    cl_kernel kernel = get_shortcut_kernel();
    cl_command_queue queue = cl.queue;

    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minw), (void*) &minw);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minh), (void*) &minh);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minc), (void*) &minc);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(sample), (void*) &sample);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(w1), (void*) &w1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(h1), (void*) &h1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c1), (void*) &c1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(add.memory), (void*) &(add.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(add.offset), (void*) &(add.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(w2), (void*) &w2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(h2), (void*) &h2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c2), (void*) &c2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(out.memory), (void*) &(out.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(out.offset), (void*) &(out.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(size,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_smooth_l1_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "smooth_l1_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void smooth_l1_gpu(int n, cl_mem_with_offset pred, cl_mem_with_offset truth, cl_mem_with_offset delta, cl_mem_with_offset error)
{

    cl_kernel kernel = get_smooth_l1_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(pred.memory), (void*) &(pred.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(pred.offset), (void*) &(pred.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth.memory), (void*) &(truth.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth.offset), (void*) &(truth.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(error.memory), (void*) &(error.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(error.offset), (void*) &(error.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(n,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_l2_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "l2_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void l2_gpu(int n, cl_mem_with_offset pred, cl_mem_with_offset truth, cl_mem_with_offset delta, cl_mem_with_offset error)
{




    cl_kernel kernel = get_l2_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(pred.memory), (void*) &(pred.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(pred.offset), (void*) &(pred.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth.memory), (void*) &(truth.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth.offset), (void*) &(truth.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(error.memory), (void*) &(error.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(error.offset), (void*) &(error.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(n,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_l1_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "l1_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void l1_gpu(int n, cl_mem_with_offset pred, cl_mem_with_offset truth, cl_mem_with_offset delta, cl_mem_with_offset error)
{




    cl_kernel kernel = get_l1_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(pred.memory), (void*) &(pred.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(pred.offset), (void*) &(pred.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth.memory), (void*) &(truth.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth.offset), (void*) &(truth.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.memory), (void*) &(delta.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta.offset), (void*) &(delta.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(error.memory), (void*) &(error.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(error.offset), (void*) &(error.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(n,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_weighted_sum_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "weighted_sum_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void weighted_sum_gpu(cl_mem_with_offset a, cl_mem_with_offset b, cl_mem_with_offset s, int num, cl_mem_with_offset c)
{




    cl_kernel kernel = get_weighted_sum_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num), (void*) &num);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a.memory), (void*) &(a.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(a.offset), (void*) &(a.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(b.memory), (void*) &(b.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(b.offset), (void*) &(b.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(s.memory), (void*) &(s.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(s.offset), (void*) &(s.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(c.memory), (void*) &(c.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(c.offset), (void*) &(c.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(num,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_weighted_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "weighted_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void weighted_delta_gpu(cl_mem_with_offset a, cl_mem_with_offset b, cl_mem_with_offset s, cl_mem_with_offset da, cl_mem_with_offset db, cl_mem_with_offset ds, int num, cl_mem_with_offset dc)
{







    cl_kernel kernel = get_weighted_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num), (void*) &num);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a.memory), (void*) &(a.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(a.offset), (void*) &(a.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(b.memory), (void*) &(b.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(b.offset), (void*) &(b.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(s.memory), (void*) &(s.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(s.offset), (void*) &(s.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(da.memory), (void*) &(da.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(da.offset), (void*) &(da.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(db.memory), (void*) &(db.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(db.offset), (void*) &(db.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(ds.memory), (void*) &(ds.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(ds.offset), (void*) &(ds.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(dc.memory), (void*) &(dc.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(dc.offset), (void*) &(dc.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(num,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_mult_add_into_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "mult_add_into_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mult_add_into_gpu(int num, cl_mem_with_offset a, cl_mem_with_offset b, cl_mem_with_offset c)
{




    cl_kernel kernel = get_mult_add_into_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num), (void*) &num);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a.memory), (void*) &(a.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(a.offset), (void*) &(a.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(b.memory), (void*) &(b.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(b.offset), (void*) &(b.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(c.memory), (void*) &(c.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(c.offset), (void*) &(c.offset));

    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(num,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_softmax_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "softmax_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void softmax_gpu(cl_mem_with_offset input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_with_offset output)
{



    cl_kernel kernel = get_softmax_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(input.memory), (void*) &(input.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(input.offset), (void*) &(input.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch_offset), (void*) &batch_offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(groups), (void*) &groups);
    cl.error = clSetKernelArg(kernel, i++, sizeof(group_offset), (void*) &group_offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(temp), (void*) &temp);
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.memory), (void*) &(output.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.offset), (void*) &(output.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(batch*groups,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


void adam_update_gpu(cl_mem_with_offset w, cl_mem_with_offset d,
    cl_mem_with_offset m, cl_mem_with_offset v,
    float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, (1-B1), d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
    fill_gpu(n, 0, d, 1);
}

cl_kernel get_scale_mask_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "scale_mask_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void scale_mask_gpu(int N, cl_mem_with_offset X, float mask_num, cl_mem_with_offset mask, float scale)
{
/* cuda code:
    scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
    check_error(cudaPeekAtLastError());
*/
//OPENCL code
//------------------------------------------------
    cl_kernel kernel = get_scale_mask_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.memory), (void*) &(X.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(X.offset), (void*) &(X.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask_num), (void*) &mask_num);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask.memory), (void*) &(mask.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask.offset), (void*) &(mask.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(scale), (void*) &scale);
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(N,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
//------------------------------------------------
}

cl_kernel get_softmax_tree_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/blas_kernels.cl", "softmax_tree_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void softmax_tree(
    cl_mem_with_offset input, int spatial,
    int batch, int stride, float temp, cl_mem_with_offset output, tree hier)
{
/*cuda code
    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
    int num = spatial*batch*hier.groups;
    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
    check_error(cudaPeekAtLastError());
    cuda_free((float *)tree_groups_size);
    cuda_free((float *)tree_groups_offset);
*/
//OPENCL code
//-------------------------------
    cl_mem_with_offset tree_groups_size = cl_make_int_array(hier.group_size, hier.groups);
    cl_mem_with_offset tree_groups_offset = cl_make_int_array(hier.group_offset, hier.groups);
    int num = spatial*batch*hier.groups;

    cl_kernel kernel = get_softmax_tree_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(input.memory), (void*) &(input.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(input.offset), (void*) &(input.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(temp), (void*) &temp);
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.memory), (void*) &(output.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(output.offset), (void*) &(output.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(hier.groups), (void*) &hier.groups);
    cl.error = clSetKernelArg(kernel, i++, sizeof(tree_groups_size.memory), (void*) &(tree_groups_size.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(tree_groups_size.offset), (void*) &(tree_groups_size.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(tree_groups_offset.memory), (void*) &(tree_groups_offset.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(tree_groups_offset.offset), (void*) &(tree_groups_offset.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(num,BLOCK)};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
    cl_free(tree_groups_size);
    cl_free(tree_groups_offset);
//-------------------------------
}

#endif

