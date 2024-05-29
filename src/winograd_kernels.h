#define BLOCK_SIZE 8
#define KERNEL_SIZE 3

#include <string.h>
static inline void winograd_f6k3_kernel_transform(
	const float g0, const float g1, const float g2,
	float transform0[restrict static 1],
	float transform1[restrict static 1],
	float transform2[restrict static 1],
	float transform3[restrict static 1],
	float transform4[restrict static 1],
	float transform5[restrict static 1],
	float transform6[restrict static 1],
	float transform7[restrict static 1],
	int rescale_coefficients)
{
	/*
	 * w0 = g0
	 * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
	 * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
	 * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
	 * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
	 * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
	 * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
	 * w7 = g2
	 */

	/*
	 * Compute
	 *   w2 := g0 + g2
	 *   w4 := g0 + 4 * g2
	 *   w6 := g2 + 4 * g0
	 */
	const float const_4 = 4.0f;
	float w2 = g0 + g2;
	float w4 = g0 + const_4 * g2;
	float w6 = g2 + const_4 * g0;

	/*
	 * Compute
	 *   w1 = (g0 + g2) + g1
	 *   w2 = (g0 + g2) - g1
	 *   w3 = (g0 + 4 * g2) + 2 * g1
	 *   w4 = (g0 + 4 * g2) - 2 * g1
	 *   w5 = (g2 + 4 * g0) + 2 * g1
	 *   w6 = (g2 + 4 * g0) - 2 * g1
	 */
	const float two_g1 = g1 * 2.0f;
	float w1 = w2 + g1;
	w2 = w2 - g1;
	float w3 = w4 + two_g1;
	w4 = w4 - two_g1;
	float w5 = w6 + two_g1;
	w6 = w6 - two_g1;

	if (rescale_coefficients) {
		const float minus_2_over_9 = -0x1.C71C72p-3f;
		w1 *= minus_2_over_9;
		w2 *= minus_2_over_9;

		const float rcp_90 = 0x1.6C16C2p-7f;
		w3 *= rcp_90;
		w4 *= rcp_90;

		const float rcp_180 = 0x1.6C16C2p-8f;
		w5 *= rcp_180;
		w6 *= rcp_180;
	}

	*transform0 = g0;
	*transform1 = w1;
	*transform2 = w2;
	*transform3 = w3;
	*transform4 = w4;
	*transform5 = w5;
	*transform6 = w6;
	*transform7 = g2;
}

static inline void winograd_f6k3_output_transform(
	const float m0, const float m1, const float m2, const float m3, const float m4, const float m5, const float m6, const float m7,
	float output0[restrict static 1],
	float output1[restrict static 1],
	float output2[restrict static 1],
	float output3[restrict static 1],
	float output4[restrict static 1],
	float output5[restrict static 1])
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	const float m1_add_m2 = m1 + m2;
	const float m1_sub_m2 = m1 - m2;
	const float m3_add_m4 = m3 + m4;
	const float m3_sub_m4 = m3 - m4;
	const float m5_add_m6 = m5 + m6;
	const float m5_sub_m6 = m5 - m6;

	float s0 = m0 + m1_add_m2;
	float s5 = m7 + m1_sub_m2;

	const float const_16 = 16.0f;
	float s1 = m1_sub_m2 + const_16 * m5_sub_m6;
	float s4 = m1_add_m2 + const_16 * m3_add_m4;

	const float const_8 = 8.0f;
	float s2 = m1_add_m2 + const_8 * m5_add_m6;
	float s3 = m1_sub_m2 + const_8 * m3_sub_m4;

	const float const_32 = 32.0f;
	s0 += const_32 * m5_add_m6;
	s5 += const_32 * m3_sub_m4;

	s0 += m3_add_m4;
	s5 += m5_sub_m6;

	const float const_2 = 2.0f;
	s1 += m3_sub_m4 * const_2;
	s4 += m5_add_m6 * const_2;

	const float const_4 = 4.0f;
	s2 += m3_add_m4 * const_4;
	s3 += m5_sub_m6 * const_4;

	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
}



inline void nnp_kwt8x8_3x3__scalar(
	const float g[restrict static 9],
	float transform[restrict static 1],
	int stride_g, int transform_stride,
	int row_count, int column_count,
	int row_offset, int column_offset)
{
	transform_stride /= sizeof(float);

	float block[KERNEL_SIZE][BLOCK_SIZE];

	for (int row = 0; row < KERNEL_SIZE; row++) {
		float w0, w1, w2, w3, w4, w5, w6, w7;
		winograd_f6k3_kernel_transform(
			g[0],
			g[1],
			g[2],
			&block[row][0], &block[row][1], &block[row][2], &block[row][3],
			&block[row][4], &block[row][5], &block[row][6], &block[row][7],
			1);
		g += KERNEL_SIZE;
	}

	for (int column = 0; column < BLOCK_SIZE; column++) {
		float w0, w1, w2, w3, w4, w5, w6, w7;
		winograd_f6k3_kernel_transform(
			block[0][column], block[1][column], block[2][column],
			&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
			1);
		*transform = w0;
		transform += transform_stride;
		*transform = w1;
		transform += transform_stride;
		*transform = w2;
		transform += transform_stride;
		*transform = w3;
		transform += transform_stride;
		*transform = w4;
		transform += transform_stride;
		*transform = w5;
		transform += transform_stride;
		*transform = w6;
		transform += transform_stride;
		*transform = w7;
		transform += transform_stride;
	}
}

static inline void winograd_f6k3_input_transform(
	const float d0, const float d1, const float d2, const float d3,
	const float d4, const float d5, const float d6, const float d7,
	float transform0[restrict static 1],
	float transform1[restrict static 1],
	float transform2[restrict static 1],
	float transform3[restrict static 1],
	float transform4[restrict static 1],
	float transform5[restrict static 1],
	float transform6[restrict static 1],
	float transform7[restrict static 1])
{
	const float const_0_25 = 0.25f;

	// Compute wd0 := d0 - d6
	float wd0 = d0 - d6;
	const float d4_sub_d2 = d4 - d2;
	// Compute wd7 := d7 - d1
	float wd7 = d7 - d1;
	const float d3_sub_d5 = d3 - d5;
	// Compute wd1 := d2 + d6
	float wd1 = d2 + d6;
	// Compute wd2 := d1 + d5
	float wd2 = d1 + d5;
	// Compute wd4 := d5 + 0.25 * d1
	float wd4 = d5 + const_0_25 * d1;
	// Compute wd5 := d6 - 5.0 * d4
	float wd5 = d6 - 5.0f * d4;
	// Compute wd3 := d6 + 0.25 * d2
	float wd3 = d6 + const_0_25 * d2;
	// Compute wd6 := d1 + 0.25 * d5
	float wd6 = d1 + const_0_25 * d5;

	const float const_5_25 = 5.25f;
	// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
	wd0 += const_5_25 * d4_sub_d2;
	// Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
	wd7 += const_5_25 * d3_sub_d5;

	const float const_4_25 = 4.25f;
	// Compute
	//   wd1 := (d6 + d2) - 4.25 * d4
	//   wd2 := (d1 + d5) - 4.25 * d3
	wd1 -= const_4_25 * d4;
	wd2 -= const_4_25 * d3;

	const float const_1_25 = 1.25f;
	// Compute
	//   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
	//   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
	//   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
	//   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
	wd3 -= const_1_25 * d4;
	const float d3_times_1_25 = d3 * const_1_25;
	wd5 += 4.0f * d2;
	wd4 -= d3_times_1_25;
	wd6 -= d3_times_1_25;

	const float const_2 = 2.0f;
	wd4 *= const_2;
	wd6 *= const_2;

	*transform0 = wd0;
	*transform1 = wd1 + wd2;
	*transform2 = wd1 - wd2;
	*transform3 = wd3 + wd4;
	*transform4 = wd3 - wd4;
	*transform5 = wd5 + wd6;
	*transform6 = wd5 - wd6;
	*transform7 = wd7;
}

void nnp_iwt8x8_3x3_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	int data_stride, int transform_stride,
	int row_count, int column_count,
	int row_offset, int column_offset)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];
	if (row_offset != 0) {
		memset(&block[0][0], 0, row_offset * BLOCK_SIZE * sizeof(float));
	}
	const int row_end = row_offset + row_count;
	if (row_end != BLOCK_SIZE) {
		memset(&block[row_end][0], 0, (BLOCK_SIZE - row_end) * BLOCK_SIZE * sizeof(float));
	}

	for (int row = row_offset; row < row_end; row++) {
		float d0, d1, d2, d3, d4, d5, d6, d7;
		d0 = d1 = d2 = d3 = d4 = d5 = d6 = d7 = 0.0f;

		const float *restrict row_data = data;
		int remaining_column_count = column_count;
		switch (column_offset) {
			case 0:
				d0 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 1:
				d1 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 2:
				d2 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 3:
				d3 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 4:
				d4 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 5:
				d5 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 6:
				d6 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 7:
				d7 = *row_data;
				break;
			default:
				-1;
        }
		winograd_f6k3_input_transform(d0, d1, d2, d3, d4, d5, d6, d7,
			&block[row][0], &block[row][1], &block[row][2], &block[row][3],
			&block[row][4], &block[row][5], &block[row][6], &block[row][7]);

		data += data_stride;
	}

	for (int column = 0; column < BLOCK_SIZE; column++) {
		float wd0, wd1, wd2, wd3, wd4, wd5, wd6, wd7;
		winograd_f6k3_input_transform(
			block[0][column], block[1][column], block[2][column], block[3][column],
			block[4][column], block[5][column], block[6][column], block[7][column],
			&wd0, &wd1, &wd2, &wd3, &wd4, &wd5, &wd6, &wd7);
		*transform = wd0;
		transform += transform_stride;
		*transform = wd1;
		transform += transform_stride;
		*transform = wd2;
		transform += transform_stride;
		*transform = wd3;
		transform += transform_stride;
		*transform = wd4;
		transform += transform_stride;
		*transform = wd5;
		transform += transform_stride;
		*transform = wd6;
		transform += transform_stride;
		*transform = wd7;
		transform += transform_stride;
	}
}


static inline void nnp_sgemm_only_3x3__scalar(int k, int update, const float* a, const float* b, float* c, int row_stride_c) {
	float acc00, acc01, acc02, acc10, acc11, acc12, acc20, acc21, acc22;
	acc00 = acc01 = acc02 = acc10 = acc11 = acc12 = acc20 = acc21 = acc22 = 0.0f;
	do {
		const float b0 = b[0];
		const float b1 = b[1];
		const float b2 = b[2];
		b += 3;

		const float a0 = a[0];
		acc00 += a0 * b0;
		acc01 += a0 * b1;
		acc02 += a0 * b2;

		const float a1 = a[1];
		acc10 += a1 * b0;
		acc11 += a1 * b1;
		acc12 += a1 * b2;

		const float a2 = a[2];
		acc20 += a2 * b0;
		acc21 += a2 * b1;
		acc22 += a2 * b2;

		a += 3;
	} while (--k);

	if (update) {
		c[0] += acc00;
		c[1] += acc01;
		c[2] += acc02;
		c += row_stride_c;
		c[0] += acc10;
		c[1] += acc11;
		c[2] += acc12;
		c += row_stride_c;
		c[0] += acc20;
		c[1] += acc21;
		c[2] += acc22;
	} else {
		c[0] = acc00;
		c[1] = acc01;
		c[2] = acc02;
		c += row_stride_c;
		c[0] = acc10;
		c[1] = acc11;
		c[2] = acc12;
		c += row_stride_c;
		c[0] = acc20;
		c[1] = acc21;
		c[2] = acc22;
	}
}

static inline void nnp_sgemm_upto_3x3__scalar(int mr, int nr, int k, int update, const float* a, const float* b, float* c, int row_stride_c) {
	float acc00, acc01, acc02, acc10, acc11, acc12, acc20, acc21, acc22;
	acc00 = acc01 = acc02 = acc10 = acc11 = acc12 = acc20 = acc21 = acc22 = 0.0f;
	do {
		float b0, b1, b2;

		b0 = *b++;
		if (nr > 1) {
			b1 = *b++;
			if (nr > 2) {
				b2 = *b++;
			}
		}

		const float a0 = *a++;
		acc00 += a0 * b0;
		acc01 += a0 * b1;
		acc02 += a0 * b2;

		if (mr > 1) {
			const float a1 = *a++;
			acc10 += a1 * b0;
			acc11 += a1 * b1;
			acc12 += a1 * b2;

			if (mr > 2) {
				const float a2 = *a++;
				acc20 += a2 * b0;
				acc21 += a2 * b1;
				acc22 += a2 * b2;
    		}
		}
	} while (--k);

	if (update) {
		switch (nr) {
			case 1:
				c[0] += acc00;
				if (mr > 1) {
					c += row_stride_c;
					c[0] += acc10;
					if (mr > 2) {
						c += row_stride_c;
						c[0] += acc20;
					}
				}				
				break;
			case 2:
				c[0] += acc00;
				c[1] += acc01;
				if (mr > 1) {
					c += row_stride_c;
					c[0] += acc10;
					c[1] += acc11;
					if (mr > 2) {
						c += row_stride_c;
						c[0] += acc20;
						c[1] += acc21;
					}
				}
				break;
			case 3:
				c[0] += acc00;
				c[1] += acc01;
				c[2] += acc02;
				if (mr > 1) {
					c += row_stride_c;
					c[0] += acc10;
					c[1] += acc11;
					c[2] += acc12;
					if (mr > 2) {
						c += row_stride_c;
						c[0] += acc20;
						c[1] += acc21;
						c[2] += acc22;
					}
				}
				break;
			default:
                -1;
		}
	} else {
		switch (nr) {
			case 1:
				c[0] = acc00;
				if (mr > 1) {
					c += row_stride_c;
					c[0] = acc10;
					if (mr > 2) {
						c += row_stride_c;
						c[0] = acc20;
					}
				}
				break;
			case 2:
				c[0] = acc00;
				c[1] = acc01;
				if (mr > 1) {
					c += row_stride_c;
					c[0] = acc10;
					c[1] = acc11;
					if (mr > 2) {
						c += row_stride_c;
						c[0] = acc20;
						c[1] = acc21;
					}
				}
				break;
			case 3:
				c[0] = acc00;
				c[1] = acc01;
				c[2] = acc02;
				if (mr > 1) {
					c += row_stride_c;
					c[0] = acc10;
					c[1] = acc11;
					c[2] = acc12;
					if (mr > 2) {
						c += row_stride_c;
						c[0] = acc20;
						c[1] = acc21;
						c[2] = acc22;
					}
				}
				break;
			default:
                -1;
		}
	}
}

