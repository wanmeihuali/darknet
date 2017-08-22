//#include "cuda_runtime.h"
//#include "curand.h"
//#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
//#include "cuda.h"
#incldue "opencl.h"
}

__kernel void binarize_kernel(__global float *x, int x_offset, int n, __global float *binary, int binary_offset)
{
    x = x + x_offset;
    binary = binary + binary_offset;
    int i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}
/*
void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}
*/
__kernel void binarize_input_kernel(__global float *input, int input_offset, int n, int size, __global float *binary, int binary_offset)
{
    input = input + input_offset;
    binary = binary + binary_offset;
    int s = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}
/*
void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}
*/

__kernel void binarize_weights_kernel(__global float *weights, int weights_offset, int n, int size, __global float *binary int binary_offset)
{
    weights = weights + weights_offset;
    binary = binary + binary_offset;
    int f = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}
/*
void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}
*/


__kernel void smooth_kernel(__global float *x, int x_offset, int n, int w, int h, int c, int size, float rate, __global float *delta, int delta_offset)
{
    x = x + x_offset;
    delta = delta + delta_offset;
    int id = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.);
    int h_offset = -(size/2.);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}
/*
extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}
*/


