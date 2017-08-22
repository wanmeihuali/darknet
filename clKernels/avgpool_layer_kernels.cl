/*typedef layer avgpool_layer;
#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif
*/
/*
extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}
*/

__kernel void forward_avgpool_layer_kernel(int n, int w, int h, int c, __global float *input, int input_offset, __global float *output, int output_offset)
{
    int id = get_global_id(0);
    input = input + input_offset;
    output = output + output_offset;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__kernel void backward_avgpool_layer_kernel(int n, int w, int h, int c, __global float *in_delta, int in_delta_offset, __global float *out_delta, int out_delta_offset)
{
    int id = get_global_id(0);
    in_delta = in_delta + in_delta_offset;
    out_delta = out_delta + out_delta_offset;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

/*
extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(cudaPeekAtLastError());
}
*/
