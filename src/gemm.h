#ifndef GEMM_H
#define GEMM_H

void gemm_bin(int M, int N, int K, float ALPHA,
              char  *A, int lda,
              float *B, int ldb,
              float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc);

#ifdef CUDA
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A_gpu, int lda,
              float *B_gpu, int ldb,
              float BETA,
              float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc);
#endif
#ifdef OPENCL
#include "openclutils.h"
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              cl_mem_with_offset A_gpu, int lda,
              cl_mem_with_offset B_gpu, int ldb,
              float BETA,
              cl_mem_with_offset C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              cl_mem_with_offset A, int lda,
              cl_mem_with_offset B, int ldb,
              float BETA,
              cl_mem_with_offset C, int ldc);
#endif
#endif
