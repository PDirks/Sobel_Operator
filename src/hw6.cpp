/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 6
 * hw6.cpp
 * May 1, 2016
 */

#include "../include/util.cuh"

/*
 * input:
 *      ./hw6 [input file name] [output file name]
 * example:
 *      ./hw5 ../data/lena.pgm ../data/out.pgm
 */
int main(int argc, char *argv[]){
    /*
     * Check input parameters
     */
    if( argc != 3){
        err( "INSUFFICIENT PARAMETERS, format: [int filter size] [input file name] [output file name]" );
    }
    std::string input_file    = argv[1];
    std::string output_file   = argv[2];

    /*
    * Load input file
    */
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint8_t *host_image = NULL;
    if(sdkLoadPGM( input_file.c_str(), &host_image, &width, &height ) == false){
        err( "ERROR ON PPM LOAD" );
    }
    uint8_t *output = NULL;

    /*
     * start timer
     */
    filter.timerStart();
    double copy_compute_time = 0;

    /*
     * Push file data to device memory & run sobel filter
     */
    sobel_filter sobel;
    copy_compute_time = sobel.gpu_load( &host_image, width, height, &output );

    /*
     * verify errors
     */
    double errors = sobel.cpu_filter_error(&host_image, output, width, height)
    if( errors < 0 ){
        err( "ERROR ON CPU IMAGE CREATE" );
    }

    /*
     * save gpu-produced image
     */
    if(sdkSavePGM( output_file.c_str(), output, width, height ) == false){
        err( "ERROR ON PPM save" );
    }

    /*
     * Print timings
     */
    double total_time = sobel.getTime();
    std::cout << "Copy compute time: " << copy_compute_time << " ms" << std::endl;
    std::cout << "Total time: " << total_time << " ms" << std::endl;
    std::cout << "Error percentage: " << errors << std::endl;

    /*
     * Cleanup
     */
    filter.timerStop();
    free(host_image);
    free(output);
    return 0;
}