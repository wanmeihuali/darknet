#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

void const_cpu(int N, float ALPHA, float *X, int INCX);

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);

#ifdef CUDA
#include "cuda.h"
#include "tree.h"
void constrain_gpu(int N, float ALPHA, float * X, int INCX);

void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, float * X, int INCX);
void supp_gpu(int N, float ALPHA, float * X, int INCX);
void mask_gpu(int N, float * X, float mask_num, float * mask);
void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
void const_gpu(int N, float ALPHA, float *X, int INCX);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);

void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);

void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
void mult_add_into_gpu(int num, float *a, float *b, float *c);
void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);

void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);

void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);
void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);

#endif
#ifdef OPENCL
#include "openclutils.h"
#include "tree.h"

void constrain_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX);
void axpy_gpu(int N, float ALPHA, cl_mem_with_offset  X, int INCX, cl_mem_with_offset  Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, cl_mem_with_offset  X, int OFFX, int INCX, cl_mem_with_offset  Y, int OFFY, int INCY);
void copy_gpu(int N, cl_mem_with_offset  X, int INCX, cl_mem_with_offset  Y, int INCY);
void copy_gpu_offset(int N, cl_mem_with_offset  X, int OFFX, int INCX, cl_mem_with_offset  Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, cl_mem_with_offset  X, int INCX);
void supp_gpu(int N, float ALPHA, cl_mem_with_offset  X, int INCX);
void mask_gpu(int N, cl_mem_with_offset  X, float mask_num, cl_mem_with_offset  mask);
void scale_mask_gpu(int N, cl_mem_with_offset  X, float mask_num, cl_mem_with_offset  mask, float scale);
void const_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX);
void pow_gpu(int N, float ALPHA, cl_mem_with_offset X, int INCX, cl_mem_with_offset Y, int INCY);
void mul_gpu(int N, cl_mem_with_offset X, int INCX, cl_mem_with_offset Y, int INCY);

void mean_gpu(cl_mem_with_offset x, int batch, int filters, int spatial, cl_mem_with_offset mean);
void variance_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, int batch, int filters, int spatial, cl_mem_with_offset variance);
void normalize_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, cl_mem_with_offset variance, int batch, int filters, int spatial);

void normalize_delta_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, cl_mem_with_offset variance, cl_mem_with_offset mean_delta, cl_mem_with_offset variance_delta, int batch, int filters, int spatial, cl_mem_with_offset delta);

void fast_mean_delta_gpu(cl_mem_with_offset delta, cl_mem_with_offset variance, int batch, int filters, int spatial, cl_mem_with_offset mean_delta);
void fast_variance_delta_gpu(cl_mem_with_offset x, cl_mem_with_offset delta, cl_mem_with_offset mean, cl_mem_with_offset variance, int batch, int filters, int spatial, cl_mem_with_offset variance_delta);

void fast_variance_gpu(cl_mem_with_offset x, cl_mem_with_offset mean, int batch, int filters, int spatial, cl_mem_with_offset variance);
void fast_mean_gpu(cl_mem_with_offset x, int batch, int filters, int spatial, cl_mem_with_offset mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem_with_offset add, int w2, int h2, int c2, cl_mem_with_offset out);
void scale_bias_gpu(cl_mem_with_offset output, cl_mem_with_offset biases, int batch, int n, int size);
void backward_scale_gpu(cl_mem_with_offset x_norm, cl_mem_with_offset delta, int batch, int n, int size, cl_mem_with_offset scale_updates);
void scale_bias_gpu(cl_mem_with_offset output, cl_mem_with_offset biases, int batch, int n, int size);
void add_bias_gpu(cl_mem_with_offset output, cl_mem_with_offset biases, int batch, int n, int size);
void backward_bias_gpu(cl_mem_with_offset bias_updates, cl_mem_with_offset delta, int batch, int n, int size);

void smooth_l1_gpu(int n, cl_mem_with_offset pred, cl_mem_with_offset truth, cl_mem_with_offset delta, cl_mem_with_offset error);
void l2_gpu(int n, cl_mem_with_offset pred, cl_mem_with_offset truth, cl_mem_with_offset delta, cl_mem_with_offset error);
void l1_gpu(int n, cl_mem_with_offset pred, cl_mem_with_offset truth, cl_mem_with_offset delta, cl_mem_with_offset error);
void weighted_delta_gpu(cl_mem_with_offset a, cl_mem_with_offset b, cl_mem_with_offset s, cl_mem_with_offset da, cl_mem_with_offset db, cl_mem_with_offset ds, int num, cl_mem_with_offset dc);
void weighted_sum_gpu(cl_mem_with_offset a, cl_mem_with_offset b, cl_mem_with_offset s, int num, cl_mem_with_offset c);
void mult_add_into_gpu(int num, cl_mem_with_offset a, cl_mem_with_offset b, cl_mem_with_offset c);
void inter_gpu(int NX, cl_mem_with_offset X, int NY, cl_mem_with_offset Y, int B, cl_mem_with_offset OUT);
void deinter_gpu(int NX, cl_mem_with_offset X, int NY, cl_mem_with_offset Y, int B, cl_mem_with_offset OUT);

void reorg_gpu(cl_mem_with_offset x, int w, int h, int c, int batch, int stride, int forward, cl_mem_with_offset out);

void softmax_gpu(cl_mem_with_offset input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_with_offset output);
void adam_update_gpu(cl_mem_with_offset w, cl_mem_with_offset d, cl_mem_with_offset m, cl_mem_with_offset v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, cl_mem_with_offset x, cl_mem_with_offset m, cl_mem_with_offset v, float B1, float B2, float rate, float eps, int t);

void flatten_gpu(cl_mem_with_offset x, int spatial, int layers, int batch, int forward, cl_mem_with_offset out);
void softmax_tree(cl_mem_with_offset input, int spatial, int batch, int stride, float temp, cl_mem_with_offset output, tree hier);
#endif
#endif // BLAS_H
