//OPENCL done
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#include <stdio.h>
#include <time.h>

static size_t get_workspace_size(layer l)
{
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}


layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.nweights = c*n*size*size;
    l.nbiases = n;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    float scale = .02;
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    for(i = 0; i < n; ++i)
    {
        l.biases[i] = 0;
    }
    l.pad = padding;

    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize)
    {
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam)
    {
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef CUDA
    l.forward_gpu = forward_deconvolutional_layer_gpu;
    l.backward_gpu = backward_deconvolutional_layer_gpu;
    l.update_gpu = update_deconvolutional_layer_gpu;

    if(gpu_index >= 0)
    {

        if (adam)
        {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

        if(batch_normalize)
        {
            l.mean_gpu = cuda_make_array(0, n);
            l.variance_gpu = cuda_make_array(0, n);

            l.rolling_mean_gpu = cuda_make_array(0, n);
            l.rolling_variance_gpu = cuda_make_array(0, n);

            l.mean_delta_gpu = cuda_make_array(0, n);
            l.variance_delta_gpu = cuda_make_array(0, n);

            l.scales_gpu = cuda_make_array(0, n);
            l.scale_updates_gpu = cuda_make_array(0, n);

            l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
            l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
        }
    }
#ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);
#endif
#endif
#ifdef OPENCL
    l.forward_gpu = forward_deconvolutional_layer_gpu;
    l.backward_gpu = backward_deconvolutional_layer_gpu;
    l.update_gpu = update_deconvolutional_layer_gpu;

    if(gpu_index >= 0)
    {

        if (adam)
        {
            l.m_gpu = cl_make_array(l.m, c*n*size*size);
            l.v_gpu = cl_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cl_make_array(l.bias_m, n);
            l.bias_v_gpu = cl_make_array(l.bias_v, n);
            l.scale_m_gpu = cl_make_array(l.scale_m, n);
            l.scale_v_gpu = cl_make_array(l.scale_v, n);
        }
        l.weights_gpu = cl_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cl_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cl_make_array(l.biases, n);
        l.bias_updates_gpu = cl_make_array(l.bias_updates, n);

        l.delta_gpu = cl_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
        l.output_gpu = cl_make_array(l.output, l.batch*l.out_h*l.out_w*n);

        if(batch_normalize)
        {
            l.mean_gpu = cl_make_array(0, n);
            l.variance_gpu = cl_make_array(0, n);

            l.rolling_mean_gpu = cl_make_array(0, n);
            l.rolling_variance_gpu = cl_make_array(0, n);

            l.mean_delta_gpu = cl_make_array(0, n);
            l.variance_delta_gpu = cl_make_array(0, n);

            l.scales_gpu = cl_make_array(0, n);
            l.scale_updates_gpu = cl_make_array(0, n);

            l.x_gpu = cl_make_array(0, l.batch*l.out_h*l.out_w*n);
            l.x_norm_gpu = cl_make_array(0, l.batch*l.out_h*l.out_w*n);
        }
    }
#endif

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_deconvolutional_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i)
    {
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j)
        {
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize)
    {
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef CUDA
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize)
    {
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
#endif
#endif
#ifdef OPENCL
    cl_free(l->delta_gpu);
    cl_free(l->output_gpu);

    l->delta_gpu =  cl_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cl_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize)
    {
        cl_free(l->x_gpu);
        cl_free(l->x_norm_gpu);

        l->x_gpu = cl_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cl_make_array(l->output, l->batch*l->outputs);
    }
#endif
    l->workspace_size = get_workspace_size(*l);
}

void forward_deconvolutional_layer(const layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i)
    {
        float *a = l.weights;
        float *b = net.input + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
    }
    if (l.batch_normalize)
    {
        forward_batchnorm_layer(l, net);
    }
    else
    {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer(layer l, network net)
{
    int i;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize)
    {
        backward_batchnorm_layer(l, net);
    }
    else
    {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i)
    {
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = net.input + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        im2col_cpu(l.delta + i*l.outputs, l.out_c, l.out_h, l.out_w,
                   l.size, l.stride, l.pad, b);
        gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta)
        {
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = net.workspace;
            float *c = net.delta + i*n*m;

            gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void update_deconvolutional_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales)
    {
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}

#ifdef OPENCL
void forward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    for(i = 0; i < l.batch; ++i){
        cl_mem_with_offset a = l.weights_gpu;
        cl_mem_with_offset b = net.input_gpu;
        b.offset += i*l.c*l.h*l.w;
        cl_mem_with_offset c = net.workspace_gpu;

        gemm_gpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_gpu(net.workspace_gpu, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu);
        l.output_gpu.offset += l.outputs;
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        cl_mem_with_offset a = net.input_gpu;
        a.offset += i*m*k;
        cl_mem_with_offset b = net.workspace_gpu;
        cl_mem_with_offset c = l.weight_updates_gpu;

        im2col_gpu(l.delta_gpu, l.out_c, l.out_h, l.out_w,
                l.size, l.stride, l.pad, b);
        l.delta_gpu.offset += l.outputs;
        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta_gpu.exist){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            cl_mem_with_offset a = l.weights_gpu;
            cl_mem_with_offset b = net.workspace_gpu;
            cl_mem_with_offset c = net.delta_gpu;
            c.offset += i*n*m;

            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void pull_deconvolutional_layer(layer l)
{
    cl_read_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cl_read_array(l.biases_gpu, l.biases, l.n);
    cl_read_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cl_read_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cl_read_array(l.scales_gpu, l.scales, l.n);
        cl_read_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cl_read_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_deconvolutional_layer(layer l)
{
    cl_write_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cl_write_array(l.biases_gpu, l.biases, l.n);
    cl_write_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cl_write_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cl_write_array(l.scales_gpu, l.scales, l.n);
        cl_write_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cl_write_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_deconvolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, size, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu.exist){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(size, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu.exist){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}
#endif // OPENCL

