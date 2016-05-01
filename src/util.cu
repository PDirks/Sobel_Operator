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

double sobel_filter::gpu_load( uint8_t **host_image, uint32_t width, uint32_t height, uint8_t** output ){

    return 0;
}// end gpu_load()

void sobel_filter::cpu_filter( uint8_t *host_image, uint8_t *cpu_image, uint32_t width, uint32_t height ){

    return;
}// end cpu_filter()

double sobel_filter::cpu_filter_error( uint8_t **host_image, uint8_t *gpu_image, uint32_t width, uint32_t height ){

    return 0;
}// end cpu_filter_error()

void sobel_filter::timer_start(){
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    return;
}// end timer_start()

void sobel_filter::timer_stop(){
    sdkStopTimer(&timer);
    sdkDeleteTimer(&timer);
    return;
}// end timer_stop()

double sobel_filter::get_time(){
    return sdkGetTimerValue(&timer);
}// end get_time()