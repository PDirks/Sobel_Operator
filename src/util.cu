/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 6
 * util.cu
 * May 1, 2016
 */

#include "../include/util.cuh"

__global__ void sobel_filter(uint8_t *input, const uint32_t width, const uint32_t height){

    return;
}// end sobel_filter()

double edge_detect::gpu_load( uint8_t **host_image, uint32_t width, uint32_t height, uint8_t** output ){
    unsigned int size = width * height * sizeof(uint8_t);

    // Alloc device
    uint8_t* device_image = NULL;

    *output = (uint8_t *)malloc(size);
    if( output == NULL ){
        err("output malloc error");
        return -1;
    }

    cudaMalloc((void**)&device_image, size);
    cudaMemcpy(device_image, *host_image, size, cudaMemcpyHostToDevice);

    // dim defined in ../include/util.cuh, we'll start with a 8x8x1
    dim3 dimBlock(DIM_X, DIM_Y, DIM_Z);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // run filter
    sobel_filter<<<dimGrid, dimBlock, 0>>>(device_image, width, height);

    cudaMemcpy(*output, device_image, size, cudaMemcpyDeviceToHost);

    return get_time();
}// end gpu_load()

void edge_detect::cpu_filter( uint8_t *host_image, uint8_t *cpu_image, uint32_t width, uint32_t height ){

    return;
}// end cpu_filter()

double edge_detect::cpu_filter_error( uint8_t **host_image, uint8_t *gpu_image, uint32_t width, uint32_t height ){

    return 0;
}// end cpu_filter_error()

void edge_detect::timer_start(){
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    return;
}// end timer_start()

void edge_detect::timer_stop(){
    sdkStopTimer(&timer);
    sdkDeleteTimer(&timer);
    return;
}// end timer_stop()

double edge_detect::get_time(){
    return sdkGetTimerValue(&timer);
}// end get_time()