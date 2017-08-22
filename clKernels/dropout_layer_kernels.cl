/*#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}*/

__kernel void yoloswag420blazeit360noscope(__global float *input, int input_offset, int size, __global float *rand, int rand_offset, float prob, float scale)
{
    input = input + input_offset;
    rand = rand + rand_offset;
    int id = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

/*
void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);  //NO function in cl for cuda_random. Work on this 
    
    //int i;
    //for(i = 0; i < size; ++i){
        //layer.rand[i] = rand_uniform();
    //}
    //cuda_push_array(layer.rand_gpu, layer.rand, size);
    

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
*/
