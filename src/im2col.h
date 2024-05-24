#ifndef IM2COL_H
#define IM2COL_H

#include <stdlib.h>
#if __riscv_vector_version==800
	#define BCAST(a,b) __builtin_epi_vfmv_v_f_2xf32(a,b)
	#define IBCAST(a,b) __builtin_epi_vmv_v_x_2xi32(a,b)

#else
	#define BCAST(a,b) __builtin_epi_vbroadcast_2xf32(a,b);
	#define IBCAST(a,b) __builtin_epi_vbroadcast_2xi32(a,b)
#endif

void im2col_cpu(float* data_im, float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad);

#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#endif
