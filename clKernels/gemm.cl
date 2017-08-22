/*#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
}
*/
// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__kernel void gemm_tn(int TA, int TB, int M, int N, int K, float ALPHA,
                    __global float *A, int a_off, int lda,
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc){
    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];
    A = A + a_off;
    B = B + b_off;
    C = C + c_off;
    int col = get_global_id(0);
    int row = get_global_id(1);

    int col_block = get_group_id(0);
    int row_block = get_group_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float val = 0;
    float orig = C[row*ldc + col];
    int tileSize;

    for(i = 0; i < K; i += BLOCK){

        int arow = y + i;
        int acol = x + row_block*BLOCK;

        int brow = y + i;
        int bcol = col;

        arow = (arow < K) ? arow : K-1;
        acol = (acol < M) ? acol : M-1;
        brow = (brow < K) ? brow : K-1;

        int aind = arow*lda + acol;
        //int aind = mad(arow, lda ,acol);
        int bind = brow*ldb + bcol;
        //int bind = mad(brow, ldb ,bcol);

        Asub[x][y] = A[aind];
        Bsub[y][x] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        tileSize = (i+BLOCK < K) ? BLOCK : K-i;

        for(j = 0; j < tileSize; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm_nt(int TA, int TB, int M, int N, int K, float ALPHA,
                    __global float *A, int a_off, int lda,
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc) {

    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];
    A = A + a_off;
    B = B + b_off;
    C = C + c_off;

    int col = get_global_id(0);
    int row = get_global_id(1);

    int col_block = get_group_id(0);
    int row_block = get_group_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float val = 0;
    float orig = C[row*ldc + col];

    int tileSize;

    for(i = 0; i < K; i += BLOCK){

        int arow = row;
        int acol = x + i;

        int brow = col_block*BLOCK + y;
        int bcol = x + i;

        brow = (brow < N) ? brow : N-1;
        acol = (acol < K) ? acol : K-1;
        bcol = (bcol < K) ? bcol : K-1;

        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;

        Asub[y][x] = A[aind];
        Bsub[x][y] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);
        tileSize = (i+BLOCK < K) ? BLOCK : K-i;
        for(j = 0; j < tileSize; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
                    __global float *A, int a_off, int lda,
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc) {

    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];
    A = A + a_off;
    B = B + b_off;
    C = C + c_off;

    int col = get_global_id(0);
    int row = get_global_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float orig = C[row*ldc+col];
    float val = 0;
    int tileSize;

    for(i = 0; i < K; i += BLOCK){

        int arow = row;
        int acol = x + i;

        int brow = y + i;
        int bcol = col;

        acol = (acol < K) ? acol : K-1;
        brow = (brow < K) ? brow : K-1;

        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;

        Asub[y][x] = A[aind];
        Bsub[y][x] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        tileSize = (i+BLOCK < K) ? BLOCK : K-i;

        for(j = 0; j < tileSize; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    __global float *A, int a_off, int lda,
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc) {

    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];
    A = A + a_off;
    B = B + b_off;
    C = C + c_off;
    float val = 0;

    int row_block = get_group_id(1);
    int col_block = get_group_id(0);

    int sub_row = get_local_id(1);
    int sub_col = get_local_id(0);

    int row = row_block*BLOCK + sub_row;
    int col = col_block*BLOCK + sub_col;

    int i,j;
    int tileSize;

    for(i = 0; i < K; i += BLOCK){
        int arow = row_block*BLOCK + sub_row;
        int acol = i + sub_col;

        int brow = i + sub_row;
        int bcol = col_block*BLOCK + sub_col;

        if(arow < M && acol < K)Asub[sub_row][sub_col] = TA ? A[arow + acol*lda] : A[arow*lda + acol];
        if(brow < K && bcol < N)Bsub[sub_row][sub_col] = TB ? B[brow + bcol*ldb] : B[brow*ldb + bcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        tileSize = (i+BLOCK < K) ? BLOCK : K-i;

        for(j = 0; j < tileSize; ++j){
            val += Asub[sub_row][j]*Bsub[j][sub_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < N){
        C[row*ldc+col] = ALPHA*val + BETA*C[row*ldc+col];
    }
}
/*
void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}
*/
