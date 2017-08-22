#ifndef COL2IM_H
#define COL2IM_H


void col2im_cpu(float* data_col,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_im);

#ifdef CUDA
void col2im_gpu(float *data_col,
                int channels, int height, int width,
                int ksize, int stride, int pad, float *data_im);
#endif
#ifdef OPENCL
void col2im_gpu(cl_mem_with_offset data_col,
                int channels, int height, int width,
                int ksize, int stride, int pad, cl_mem_with_offset data_im);
#endif
#endif
