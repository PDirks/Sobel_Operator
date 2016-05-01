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
     * Push file data to device memory & run sobel filter
     */

    /*
     * verify errors
     */


    /*
     * Print timings
     */

    /*
     * Cleanup
     */
    free(host_image);
    free(output);
    return 0;
}