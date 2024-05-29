#ifndef WINOGRAD_H
#define WINOGRAD_H

#include "winograd_kernels.h"
#if __riscv_vector_version==800
    #define BCAST(a,b) __builtin_epi_vfmv_v_f_2xf32(a,b)
#else
    #define BCAST(a,b) __builtin_epi_vbroadcast_2xf32(a,b);
#endif

void output_parallelize_2d_tile_2d(
    float * output, 
    void * output_transform, 
    int tuple_size, 
    int tiles_count, 
    int tiles_width_count, 
    int tiles_block_max,
    int output_channels, 
    int output_width, 
    int output_height, 
    int output_tile_size, 
    int range_i, 
    int range_j, 
    int tile_i, 
    int tile_j, 
    int tiles);

void tuple_parallelize_2d_tile_2d(
    int tuple_elements, 
    int tuple_size, 
    int tiles_subblock_max, 
    int input_channels_block_start, 
    int input_channels_block_size, 
    int output_channels, 
    int output_channels_subblock_max, 
    int output_channels_block_start, 
    void * input_transform, 
    void * kernel_transform, 
    void * output_transform,
    int range_i,
    int range_j,
    int tile_i,
    int tile_j);

void input_parallelize_2d_tile_2d_intertile(
    float * input, 
    void * input_transform,
    int tuple_size,
    int tiles_count, 
    int tiles_width_count, 
    int input_channels_block_start,
    int input_channels_block_size, 
    int input_width, 
    int input_height, 
    int pad, 
    int tile_size, 
    int tile_step,
    int range_i, 
    int range_j, 
    int tile_i, 
    int tile_j, 
    int tiles);

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
