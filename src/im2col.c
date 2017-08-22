//OPENCL done
#include "im2col.h"
#include <stdio.h>
#include "openclutils.h"

float im2col_get_pixel(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
            row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}

#ifdef OPENCL

cl_kernel get_im2col_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init)
    {
        kernel = get_kernel("clKernels/im2col_kernels.cl", "im2col_gpu_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void im2col_gpu(cl_mem_with_offset im,
         int channels, int height, int width,
         int ksize, int stride, int pad, cl_mem_with_offset data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    cl_kernel kernel = get_im2col_kernel();
    cl_command_queue queue = cl.queue;
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(num_kernels), (void*) &num_kernels);
    cl.error = clSetKernelArg(kernel, i++, sizeof(im.memory), (void*) &(im.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(im.offset), (void*) &(im.offset));
    cl.error = clSetKernelArg(kernel, i++, sizeof(height), (void*) &height);
    cl.error = clSetKernelArg(kernel, i++, sizeof(width), (void*) &width);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ksize), (void*) &ksize);
    cl.error = clSetKernelArg(kernel, i++, sizeof(pad), (void*) &pad);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(height_col), (void*) &height_col);
    cl.error = clSetKernelArg(kernel, i++, sizeof(width_col), (void*) &width_col);
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_col.memory), (void*) &(data_col.memory));
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_col.offset), (void*) &(data_col.offset));
    cl_check_error(cl);

    const size_t gsize[] = {cl_global_size(num_kernels,BLOCK)};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

#endif
