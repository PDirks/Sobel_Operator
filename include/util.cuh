/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 6
 * util.cuh
 * May 1, 2016
 */

#include "../include/sdkHelper.h"
#include <stdint.h> // uint
#include <stdio.h>
#include <cuda_runtime.h>

#define _TEST_KERNEL_CU_

// constants
#define DIM_X           8
#define DIM_Y           8
#define DIM_Z           1

#define DATA_OUT_COMPUTE    0
#define DATA_OUT_TOTAL      0
#define DATA_OUT_NORMAL     1

// debug helpers
#ifndef err
#define err(e) \
    std::cerr << BRED << "[ERROR] " << e << GREY << std::endl; \
    return 0;
#endif

#ifndef debug_err
#define debug_err(e) \
    std::cerr << BRED << "[DEBUG] " << e << GREY << std::endl;
#endif

#ifndef debug_msg
#define debug_msg(e) \
    std::cout << GREEN << "[DEBUG] " << e << GREY << std::endl;
#endif

// colors
#ifndef BLACK
#define BLACK   "\033[0;30m"
#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define BROWN   "\033[0;33m"
#define BLUE    "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN    "\033[0;36m"
#define GREY    "\033[0;37m"

#define BRED    "\033[1;31m"
#define BGREEN  "\033[1;32m"
#define BBLUE   "\033[1;34m"
#define BCYAN   "\033[1;36m"
#define BGREY   "\033[1;37m"
#endif


class sobel_filter{
public:
    StopWatchInterface *timer;
    double gpu_load( uint8_t **host_image, uint32_t width, uint32_t height, uint8_t** output );
    void cpu_filter( uint8_t *host_image, uint8_t *cpu_image, uint32_t width, uint32_t height );
    double cpu_filter_error( uint8_t **host_image, uint8_t *gpu_image, uint32_t width, uint32_t height );

    void timer_start();
    void timer_stop();
    double get_time();
};