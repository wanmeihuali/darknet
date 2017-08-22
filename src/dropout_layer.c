//OPENCL done
#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include "openclutils.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
#ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
#ifdef CUDA
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
#elif defined OPENCL
    l.rand_gpu = cl_make_array(l.rand, inputs*batch);
#endif
#endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
#ifdef CUDA
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
#elif defined OPENCL
    cl_free(l->rand_gpu);

    l->rand_gpu = cl_make_array(l->rand, inputs*l->batch);
#endif
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i)
    {
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i)
    {
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}

#ifdef OPENCL

cl_kernel get_yoloswag420blazeit360noscope_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("clKernels/dropout_layer_kernels.cl", "yoloswag420blazeit360noscope_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cl_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

    /*cuda code:
    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(
        net.input_gpu,
        size,
        layer.rand_gpu,
        layer.probability,
        layer.scale);
    check_error(cudaPeekAtLastError());
    */
//opencl code
//-------------------------------------------------------
    cl_kernel kernel = get_yoloswag420blazeit360noscope_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(net.input_gpu.memory), (void*) &(net.input_gpu.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(net.input_gpu.offset), (void*) &(net.input_gpu.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_gpu.memory), (void*) &(layer.rand_gpu.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_gpu.offset), (void*) &(layer.rand_gpu.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.probability), (void*) &layer.probability);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.scale), (void*) &layer.scale);
    cl_check_error(cl);

    size_t gsize[] = {cl_global_size(size, BLOCK)};
    size_t localws[]={BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
//-------------------------------------------------------


}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu.exist) return;
    int size = layer.inputs*layer.batch;

    /*cuda code:
    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(
    net.delta_gpu,
    size,
    layer.rand_gpu,
    layer.probability,
    layer.scale);
    check_error(cudaPeekAtLastError());
    */
//opencl code
//-------------------------------------------------------
    cl_kernel kernel = get_yoloswag420blazeit360noscope_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(net.delta_gpu.memory), (void*) &(net.delta_gpu.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(net.delta_gpu.offset), (void*) &(net.delta_gpu.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_gpu.memory), (void*) &(layer.rand_gpu.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_gpu.offset), (void*) &(layer.rand_gpu.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.probability), (void*) &layer.probability);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.scale), (void*) &layer.scale);
    cl_check_error(cl);

    size_t gsize[] = {cl_global_size(size, BLOCK)};
    size_t localws[]={BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
//-------------------------------------------------------
}

#endif
