#ifndef WINOGRAD_H
#define WINOGRAD_H

#if __riscv_vector_version==800
	#define BCAST(a,b) __builtin_epi_vfmv_v_f_2xf32(a,b)
#else
	#define BCAST(a,b) __builtin_epi_vbroadcast_2xf32(a,b);
#endif

void winograd(int input_channels, 
              int output_channels, 
              int input_height, 
              int output_height, 
              int pad,
              int kernel_size,
              int stride,
              const float *input,
              const float *kernel,
              float *output,
              );
/*
enum nnp_status nnp_convolution_inference(
    enum nnp_convolution_algorithm algorithm,
    enum nnp_convolution_transform_strategy transform_strategy,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_padding input_padding,
    struct nnp_size kernel_size,
    struct nnp_size output_subsampling,
    const float *input,
    const float *kernel,
    const float *bias,
    float *output,
    void *workspace_buffer,
    size_t *workspace_size,
    enum nnp_activation activation,
    const void *activation_parameters,
    pthreadpool_t threadpool,
    struct nnp_profile *profile)

            nnp_convolution_inference(
                nnp_convolution_algorithm_wt8x8,
                nnp_convolution_transform_strategy_tuple_based,
                (size_t)(l.c / l.groups),
                (size_t)m,
                input_size,
                input_padding,
                kernel_size,
                stride,
                net.input,
                l.weights + j * l.nweights / l.groups,
                NULL,
                l.output + j * n * m,
                NULL,
                NULL,
                nnp_activation_identity,
                NULL,
                net.threadpool,
                NULL);
*/
#endif
