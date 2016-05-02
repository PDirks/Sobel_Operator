/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 6
 * util.cu
 * May 1, 2016
 */

#include "../include/util.cuh"

__global__ void sobel_filter(uint8_t *input, uint8_t *output, const uint32_t width, const uint32_t height){
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x > width) || (y > height) || ((int)x < 0) || ((int)y < 0)) {
        return;
    }

    int32_t gradient[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, -1}
    };

    double dx = 0;
    double dy = 0;
    for( int32_t i = -1; i <= 1; i++ ){
        for( int32_t j = -1; j <= 1; j++ ){
             //pixel = gradient[j+1][i+1] * input[ ((y+j)*width) + (x+i) ];
            dx += input[(i+x)*width + (j+y)] + gradient[i][j];
            dy += input[(i+x)*width + (j+y)] + gradient[j][i];
        }
    }

    double total_magnitude = sqrt( dx * dx + dy * dy );
    if( total_magnitude < 0 ){
        total_magnitude = 0;
    }
    if( total_magnitude > 255 ){
        total_magnitude = 255;
    }

    output[x * width + y ] = total_magnitude;
    return;
}// end sobel_filter()

double edge_detect::gpu_load( uint8_t **host_image, uint32_t width, uint32_t height, uint8_t** output ){
    unsigned int size = width * height * sizeof(uint8_t);

    // Alloc device
    uint8_t* device_image_in    = NULL;
    uint8_t* device_image_out   = NULL;

    *output = (uint8_t *)malloc(size);
    if( output == NULL ){
        err("output malloc error");
        return -1;
    }

    cudaMalloc((void**)&device_image_in, size);
    cudaMalloc((void**)&device_image_out, size);
    cudaMemcpy(device_image_in, *host_image, size, cudaMemcpyHostToDevice);

    // dim defined in ../include/util.cuh, we'll start with a 8x8x1
    dim3 dimBlock(DIM_X, DIM_Y, DIM_Z);
    dim3 dimGrid(width / DIM_X, height / DIM_Y, 1);

    // run filter
    sobel_filter<<<dimGrid, dimBlock, 0>>>(device_image_in, device_image_out, width, height);

    cudaMemcpy(output, (void *)device_image_out, size, cudaMemcpyDeviceToHost);

    cudaFree( device_image_in );
    cudaFree( device_image_out );

    return get_time();
}// end gpu_load()

void edge_detect::cpu_filter( uint8_t *host_image, uint8_t *cpu_image, uint32_t width, uint32_t height ){

    return;
}// end cpu_filter()

double edge_detect::cpu_filter_error( uint8_t **host_image, uint8_t *gpu_image, uint32_t width, uint32_t height ){
    uint32_t window_size = height * width;

    /*
     * create cpu_image
     */
    uint8_t *cpu_image = (uint8_t*)calloc( window_size, sizeof( uint8_t ) );
    if( cpu_image == NULL ){
        return -1;
    }

    cpu_filter( *host_image, cpu_image, width, height );

    /*
     * compare pixels for errors
     */
    uint32_t error_count = 0;
    for( uint32_t i = 0; i < window_size; i++ ){
        if( cpu_image[i] != gpu_image[i] ){
            error_count++;
        }
    }

#if DEBUG
    /*
     * save cpu image file
     */
    char cpu_file[] = "out_cpu.pgm";
    if(sdkSavePGM( cpu_file, cpu_image, width, height ) == false){
        return -1;
    }
#endif

    free(cpu_image);

    return (double) (error_count / window_size);
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