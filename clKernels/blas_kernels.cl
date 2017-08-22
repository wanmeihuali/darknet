//#include "cuda_runtime.h"
//#include "curand.h"
//#include "cublas_v2.h"
//#include <assert.h>

//extern "C" {
//#include "blas.h"
//#include "cuda.h"
//#include "utils.h" //is this one useful??
//}

__kernel void scale_bias_kernel(__global float *output, int output_offset, __global float *biases, int biases_offset, int n, int size)
{
    int offset = get_global_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    int filter = get_group_id(1);//blockIdx.y;
    int batch = get_group_id(2);//blockIdx.z;
    output = output + output_offset;
    biases = biases + biases_offset;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}
/*
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void backward_scale_kernel(__global float *x_norm, int x_norm_offset, __global float *delta, int delta_offset, int batch, int n, int size, __global float *scale_updates, int scale_updates_offset)
{
    __local float part[BLOCK];
    int i,b;
    int filter = get_group_id(0);//blockIdx.x;
    int p = get_local_id(0);//threadIdx.x;
    x_norm = x_norm + x_norm_offset;
    delta = delta + delta_offset;
    scale_updates = scale_updates + scale_updates_offset;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

/*
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void add_bias_kernel(__global float *output, int output_offset, __global float *biases, int biases_offset, int n, int size)
{
    int offset = get_global_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    int filter = get_group_id(1);//blockIdx.y;
    int batch = get_group_id(2);//blockIdx.z;
    output = output + output_offset;
    biases = biases + biases_offset;

    if(offset < size) output[(batch*n+filter)*size + offset] += biases[filter];
}

/*
void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    add_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void backward_bias_kernel(__global float *bias_updates, int bias_updates_offset, __global float *delta, int delta_offset, int batch, int n, int size)
{
    __local float part[BLOCK];
    int i,b;
    int filter = get_group_id(0);//blockIdx.x;
    int p = get_local_id(0);//threadIdx.x;
    bias_updates = bias_updates + bias_updates_offset;
    delta = delta + delta_offset;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}
/*
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void adam_kernel(int N, __global float *x, int x_offset, __global float *m, int m_offset, __global float *v, int v_offset, float B1, float B2, float rate, float eps, int t)
{
    int index = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    x = x + x_offset;
    m = m + m_offset;
    v = v + v_offset;
    if (index >= N) return;

    x[index] = x[index] + (rate * sqrt(1.-pow(B2, t)) / (1.-pow(B1, t)) * m[index] / (sqrt(v[index]) + eps));
}
/*
extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    adam_kernel<<<cuda_gridsize(n), BLOCK>>>(n, x, m, v, B1, B2, rate, eps, t);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void normalize_kernel(int N, __global float *x, int x_offset, __global float *mean, int mean_offset, __global float *variance, int variance_offset, int batch, int filters, int spatial)
{
    int index = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //(get_group_id(0)+get_group_id(1)*get_num_groups(0))*get_local_size(0)+get_local_id(0);
    x = x + x_offset;
    mean = mean + mean_offset;
    variance = variance + variance_offset;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
}
/*
extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void normalize_delta_kernel(int N, __global float *x, int x_offset, __global float *mean, int mean_offset, __global float *variance, int variance_offset, __global float *mean_delta, int mean_delta_offset, __global float *variance_delta, int variance_delta_offset, int batch, int filters, int spatial, __global float *delta, int delta_offset)
{
    int index = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    x = x + x_offset;
    mean = mean + mean_offset;
    variance = variance + variance_offset;
    mean_delta = mean_delta + mean_delta_offset;
    variance_delta = variance_delta + variance_delta_offset;
    delta = delta + delta_offset;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}
/*
extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    check_error(cudaPeekAtLastError());
}
*/

//variance_delta_kernel seems no use??
__kernel void  variance_delta_kernel(__global float *x, __global float *delta, __global float *mean, __global float *variance, int batch, int filters, int spatial, __global float *variance_delta)
{
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
}

//accumulate_kernel is no use!!
__kernel void accumulate_kernel(__global float *x, int n, int groups, __global float *sum)
{
    int k;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

__kernel void fast_mean_delta_kernel(__global float *delta, int delta_offset, __global float *variance, int variance_offset, int batch, int filters, int spatial, __global float *mean_delta, int mean_delta_offset)
{
    const int threads = BLOCK;
    __local float local_[BLOCK];

    int id = get_local_id(0);//threadIdx.x;
    local_[id] = 0;

    int filter = get_group_id(0);//blockIdx.x;
    delta = delta + delta_offset;
    variance = variance + variance_offset;
    mean_delta = mean_delta + mean_delta_offset;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local_[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local_[i];
        }
        mean_delta[filter] *= (-1./sqrt(variance[filter] + .00001f));
    }
}
/*
extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

*/

__kernel void  fast_variance_delta_kernel(__global float *x, int x_offset, __global float *delta, int delta_offset, __global float *mean, int mean_offset, __global float *variance, int variance_offset, int batch, int filters, int spatial, __global float *variance_delta, int variance_delta_offset)
{
    const int threads = BLOCK;
    __local float local_[BLOCK];
    x = x + x_offset;
    delta = delta + delta_offset;
    mean = mean + mean_offset;
    variance = variance + variance_offset;
    variance_delta = variance_delta + variance_delta_offset;

    int id = get_local_id(0);//threadIdx.x;
    local_[id] = 0;

    int filter = get_group_id(0);//blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local_[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local_[i];
        }
        variance_delta[filter] *= -.5 * pow(variance[filter] + .00001f, (float)(-3./2.));
    }
}
/*
extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void mean_delta_kernel(__global float *delta, int delta_offset, __global float *variance, int variance_offset, int batch, int filters, int spatial, __global float *mean_delta, int mean_delta_offset)
{
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    delta = delta + delta_offset;
    variance = variance + variance_offset;
    mean_delta = mean_delta + mean_delta_offset;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
}
/*
extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}
*/


__kernel void  mean_kernel(__global float *x, int x_offset, int batch, int filters, int spatial, __global float *mean, int mean_offset)
{
    float scale = 1./(batch * spatial);
    x = x + x_offset;
    mean = mean + mean_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}
/*
extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void variance_kernel(__global float *x, int x_offset, __global float *mean, int mean_offset, int batch, int filters, int spatial, __global float *variance, int variance_offset)
{
    x = x + x_offset;
    mean = mean + mean_offset;
    variance = variance + variance_offset;
    float scale = 1./(batch * spatial - 1);
    int j,k;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += pow((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}
/*
extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void reorg_kernel(int N, __global float *x, int x_offset, int w, int h, int c, int batch, int stride, int forward, __global float *out, int out_offset)
{
    x = x + x_offset;
    out = out + out_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

/*
extern "C" void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int size = w*h*c*batch;
    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, stride, forward, out);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void axpy_kernel(int N, float ALPHA, __global float *X, int OFFX, int INCX,  __global float *Y, int OFFY, int INCY)
{
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

/*
extern "C" void axpy_ongpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}
*/
/*
extern "C" void axpy_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}
*/
__kernel void pow_kernel(int N, float ALPHA, __global float *X, int X_offset, int INCX, __global float *Y, int Y_offset, int INCY)
{
    X = X + X_offset;
    Y = Y + Y_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}
/*
extern "C" void pow_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void const_kernel(int N, float ALPHA, __global float *X, int X_offset, int INCX)
{
    X = X + X_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}
/*
extern "C" void const_ongpu(int N, float ALPHA, float * X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void constrain_kernel(int N, float ALPHA, __global float *X, int X_offset, int INCX)
{
    X = X + X_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = fmin(ALPHA, fmax(-ALPHA, X[i*INCX]));
}
/*
extern "C" void constrain_ongpu(int N, float ALPHA, float * X, int INCX)
{
    constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void supp_kernel(int N, float ALPHA, __global float *X, int X_offset, int INCX)
{
    X = X + X_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}
/*
extern "C" void supp_ongpu(int N, float ALPHA, float * X, int INCX)
{
    supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void add_kernel(int N, float ALPHA, __global float *X, int X_offset, int INCX)
{
    X = X + X_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] += ALPHA;
}
/*
extern "C" void add_ongpu(int N, float ALPHA, float * X, int INCX)
{
    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void scal_kernel(int N, float ALPHA, __global float *X, int x_offset, int INCX)
{
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX + x_offset] *= ALPHA;
}
/*
extern "C" void scal_ongpu(int N, float ALPHA, float * X, int INCX)
{
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void fill_kernel(int N, float ALPHA, __global float *X, int X_offset, int INCX)
{
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[X_offset+i*INCX] = ALPHA;
}
/*
extern "C" void fill_ongpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void mask_kernel(int n, __global float *x, int x_offset, float mask_num, float __global *mask, int mask_offset)
{
    x = x + x_offset;
    mask = mask + mask_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] = mask_num;
}
/*
extern "C" void mask_ongpu(int N, float * X, float mask_num, float * mask)
{
    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void copy_kernel(int N, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}
/*
extern "C" void copy_ongpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}
*/

/*extern "C" void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
}
*/
__kernel void mul_kernel(int N, __global float *X, int X_offset, int INCX, __global float *Y, int Y_offset, int INCY)
{
    X = X + X_offset;
    Y = Y + Y_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}
/*
extern "C" void mul_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void  fast_mean_kernel(__global float *x, int x_offset, int batch, int filters, int spatial, __global float *mean, int mean_offset)
{
    const int threads = BLOCK;
    __local float local_[BLOCK];
    x = x + x_offset;
    mean = mean + mean_offset;
    int id = get_local_id(0);//threadIdx.x;
    local_[id] = 0;

    int filter = get_group_id(0);//blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local_[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local_[i];
        }
        mean[filter] /= spatial * batch;
    }
}
/*
extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void  fast_variance_kernel(__global float *x, int x_offset, __global float *mean, int mean_offset, int batch, int filters, int spatial, __global float *variance, int variance_offset)
{
    x = x + x_offset;
    mean = mean + mean_offset;
    variance = variance + variance_offset;
    const int threads = BLOCK;
    __local float local_[BLOCK];

    int id = get_local_id(0);//threadIdx.x;
    local_[id] = 0;

    int filter = get_group_id(0);//blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local_[id] += (i+id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;
        }
    }

    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local_[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}
/*
extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void flatten_kernel(int N, __global float *x, int x_offset, int spatial, int layers, int batch, int forward, __global float *out, int out_offset)
{
    x = x + x_offset;
    out = out + out_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}
/*
extern "C" void flatten_ongpu(float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int size = spatial*batch*layers;
    flatten_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, spatial, layers, batch, forward, out);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, __global float *add, int add_offset, int w2, int h2, int c2, __global float *out, int out_offset)
{
    add = add + add_offset;
    out = out + out_offset;
    int id = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] += add[add_index];
}
/*
extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
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
    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void smooth_l1_kernel(int n, __global float *pred, int pred_offset, __global float *truth, int truth_offset, __global float *delta, int delta_offset, __global float *error, int error_offset)
{
    pred = pred + pred_offset;
    truth = truth + truth_offset;
    delta = delta + delta_offset;
    error = error + error_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}
/*
extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void l2_kernel(int n, __global float *pred, int pred_offset, __global float *truth, int truth_offset, __global float *delta, int delta_offset, __global float *error, int error_offset)
{
    pred = pred + pred_offset;
    truth = truth + truth_offset;
    delta = delta + delta_offset;
    error = error + error_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = diff;
    }
}
/*
extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void l1_kernel(int n, __global float *pred, int pred_offset, __global float *truth, int truth_offset, __global float *delta, int delta_offset, __global float *error, int error_offset)
{
    pred = pred + pred_offset;
    truth = truth + truth_offset;
    delta = delta + delta_offset;
    error = error + error_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}
/*
extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void weighted_sum_kernel(int n, __global float *a, int a_offset, __global float *b, int b_offset, __global float *s, int s_offset, __global float *c, int c_offset)
{
    a = a + a_offset;
    b = b + b_offset;
    s = s + s_offset;
    c = c + c_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}
/*
extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
{
    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, c);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void weighted_delta_kernel(int n, __global float *a, int a_offset, __global float *b, int b_offset, __global float *s, int s_offset, __global float *da, int da_offset, __global float *db, int db_offset, __global float *ds, int ds_offset, __global float *dc, int dc_offset)
{
    a = a + a_offset;
    b = b + b_offset;
    s = s + s_offset;
    da = da + da_offset;
    db = db + db_offset;
    ds = ds + ds_offset;
    dc = dc + dc_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * a[i] + dc[i] * -b[i];
    }
}
/*
extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
{
    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, da, db, ds, dc);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void mult_add_into_kernel(int n, __global float *a, int a_offset, __global float *b, int b_offset, __global float *c, int c_offset)
{
    a = a + a_offset;
    b = b + b_offset;
    c = c + c_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] += a[i]*b[i];
    }
}
/*
extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
{
    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, c);
    check_error(cudaPeekAtLastError());
}
*/

void softmax_device(__global float *input, int n, float temp, int stride, __global float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

__kernel void softmax_kernel(__global float *input, int input_offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, __global float *output, int output_offset)
{
    input = input + input_offset;
    output = output + output_offset;
    int id = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}
/*
extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
    check_error(cudaPeekAtLastError());
}
*/
