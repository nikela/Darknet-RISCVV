#ifndef WINOGRAD_H
#define WINOGRAD_H

#if __riscv_vector_version==800
    #define BCAST(a,b) __builtin_epi_vfmv_v_f_2xf32(a,b)
#else
    #define BCAST(a,b) __builtin_epi_vbroadcast_2xf32(a,b);
#endif


void kernel_parallelize_2d_tile_2d_intertile(
    float * kernel, 
    void * kernel_transform,
    int tuple_size,
    int input_channels, 
    int input_channels_block_size,
    int output_channels, 
    int kernel_size,
    int range_i,
    int range_j,
    int tile_i,
    int tile_j, 
    int tiles);

void winograd(
    int input_channels, 
    int output_channels, 
    int input_height, 
    int input_width, 
    int pad,
    int kernel_size,
    int stride,
    const float *input,
    const float *kernel,
    float *output);
#endif
