#ifndef IM2COL_H
#define IM2COL_H

#include "openclutils.h"

void im2col_cpu(float* data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_col);

#ifdef CUDA

void im2col_gpu(float *im,
                int channels, int height, int width,
                int ksize, int stride, int pad,float *data_col);

#endif
#ifdef OPENCL

void im2col_gpu(cl_mem_with_offset im,
                int channels, int height, int width,
                int ksize, int stride, int pad,cl_mem_with_offset data_col);

#endif
#endif
