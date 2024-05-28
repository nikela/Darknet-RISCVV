/***** Manual vectorization with 16 unroll + 8 unroll + double buffer = 32 vector register usage*///
/***********************3. loop interchange with manual vectorization with ALPHA=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_noalpha_doublebuffwith32reg(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i = 0; i < M-15; i += 16) {
        __epi_2xf32 vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7,vb8, vb9, vb10, vb11, vb12, vb13, vb14, vb15,vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;

        LOAD_C_UNROLL_16();
	// double buffer scheme implementation -start
        int flag=0;
	for ( k = 0; k < K-7; k +=8) {
		 if (flag==0){
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);
                 vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
		  vb8 = __builtin_epi_vload_2xf32(&B[(k+8)*ldb+j], gvl);
                vb9 = __builtin_epi_vload_2xf32(&B[(k+9)*ldb+j], gvl);
                vb10 = __builtin_epi_vload_2xf32(&B[(k+10)*ldb+j], gvl);
                vb11 = __builtin_epi_vload_2xf32(&B[(k+11)*ldb+j], gvl);
                 vb12 = __builtin_epi_vload_2xf32(&B[(k+12)*ldb+j], gvl);
                 vb13 = __builtin_epi_vload_2xf32(&B[(k+13)*ldb+j], gvl);
                 vb14 = __builtin_epi_vload_2xf32(&B[(k+14)*ldb+j], gvl);
                 vb15 = __builtin_epi_vload_2xf32(&B[(k+15)*ldb+j], gvl);
                }
                else
                {
			if(flag & 1)
			{
			   if(k<K-8)
			   {
                 		vb = __builtin_epi_vload_2xf32(&B[(k+8)*ldb+j], gvl);
                 		vb1 = __builtin_epi_vload_2xf32(&B[(k+9)*ldb+j], gvl);
                 		vb2 = __builtin_epi_vload_2xf32(&B[(k+10)*ldb+j], gvl);
                 		vb3 = __builtin_epi_vload_2xf32(&B[(k+11)*ldb+j], gvl);
				vb4 = __builtin_epi_vload_2xf32(&B[(k+12)*ldb+j], gvl);
                 		vb5 = __builtin_epi_vload_2xf32(&B[(k+13)*ldb+j], gvl);
                 		vb6 = __builtin_epi_vload_2xf32(&B[(k+14)*ldb+j], gvl);
                 		vb7 = __builtin_epi_vload_2xf32(&B[(k+15)*ldb+j], gvl);

                           }
                        }
                        else
			{
			    if(k<K-8)
                           {
                                vb8 = __builtin_epi_vload_2xf32(&B[(k+8)*ldb+j], gvl);
                                vb9 = __builtin_epi_vload_2xf32(&B[(k+9)*ldb+j], gvl);
                                vb10 = __builtin_epi_vload_2xf32(&B[(k+10)*ldb+j], gvl);
                                vb11 = __builtin_epi_vload_2xf32(&B[(k+11)*ldb+j], gvl);
			       vb12 = __builtin_epi_vload_2xf32(&B[(k+12)*ldb+j], gvl);
                                vb13 = __builtin_epi_vload_2xf32(&B[(k+13)*ldb+j], gvl);
                                vb14 = __builtin_epi_vload_2xf32(&B[(k+14)*ldb+j], gvl);
                                vb15 = __builtin_epi_vload_2xf32(&B[(k+15)*ldb+j], gvl);
                           }
                	}
		}

		//double buffer scheme implementation - end
		if(flag & 1)
		{

               __epi_2xf32 vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha0 = BCAST(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb8, gvl); // sum += ALPHA*A*B

		__epi_2xf32 vaalpha01 = BCAST(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha02 = BCAST(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha03 = BCAST(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha04 = BCAST(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha05 = BCAST( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha06 = BCAST(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha07 = BCAST(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha08 = BCAST(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha9= BCAST(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha09= BCAST(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha010 = BCAST(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha011 = BCAST(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha012 = BCAST(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha013 = BCAST(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha014 = BCAST(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha15 = BCAST(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha015 = BCAST(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb9, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = BCAST(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb10, gvl); // sum += ALPHA*A*B

                vaalpha0 = BCAST(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb11, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb10, gvl); // sum += ALPHA*A*B

               vaalpha01 = BCAST(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb11, gvl); // sum += ALPHA*A*B

               vaalpha2 = BCAST(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb10, gvl); // sum += ALPHA*A*B

                vaalpha02 = BCAST(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb11, gvl); // sum += ALPHA*A*B


                vaalpha3 = BCAST(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb10, gvl); // sum += ALPHA*A*B

                vaalpha03 = BCAST(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb11, gvl); // sum += ALPHA*A*B

               vaalpha4 = BCAST( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb10, gvl); // sum += ALPHA*A*B

               vaalpha04 = BCAST(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb11, gvl); // sum += ALPHA*A*B

                vaalpha5 = BCAST(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb10, gvl); // sum += ALPHA*A*B

		vaalpha05 = BCAST(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb11, gvl); // sum += ALPHA*A*B

                vaalpha6 = BCAST(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb10, gvl); // sum += ALPHA*A*B

		vaalpha06 = BCAST(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb11, gvl); // sum += ALPHA*A*B

                 vaalpha7 = BCAST(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb10, gvl); // sum += ALPHA*A*B

                  vaalpha07 = BCAST(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb11, gvl); // sum += ALPHA*A*B


                  vaalpha8 = BCAST(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb10, gvl); // sum += ALPHA*A*B

                  vaalpha08 = BCAST(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb11, gvl); // sum += ALPHA*A*B


                   vaalpha9= BCAST(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb10, gvl); // sum += ALPHA*A*B

               	 vaalpha09= BCAST(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb11, gvl); // sum += ALPHA*A*B


                 vaalpha10 = BCAST(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb10, gvl); // sum += ALPHA*A*B

                 vaalpha010 = BCAST(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb11, gvl); // sum += ALPHA*A*B

                vaalpha11 = BCAST(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb10, gvl); // sum += ALPHA*A*B

                 vaalpha011 = BCAST(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb11, gvl); // sum += ALPHA*A*B


                vaalpha12 = BCAST(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb10, gvl); // sum += ALPHA*A*B

                vaalpha012 = BCAST(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb11, gvl); // sum += ALPHA*A*B


                 vaalpha13 = BCAST(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb10, gvl); // sum += ALPHA*A*B

		vaalpha013 = BCAST( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb11, gvl); // sum += ALPHA*A*B


                   vaalpha14 = BCAST(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb10, gvl); // sum += ALPHA*A*B

                 vaalpha014 = BCAST(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb11, gvl); // sum += ALPHA*A*B


                   vaalpha15 = BCAST(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb10, gvl); // sum += ALPHA*A*B

                   vaalpha015 = BCAST(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb11, gvl); // sum += ALPHA*A*B

		/////
		/* unroll 6*/
		  vaalpha = BCAST(A[i*lda+(k+4)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb12, gvl); // sum += ALPHA*A*B

                vaalpha0 = BCAST(A[i*lda+(k+5)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb13, gvl); // sum += ALPHA*A*B

		  vaalpha1 = BCAST(A[(i+1)*lda+(k+4)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb12, gvl); // sum += ALPHA*A*B

	 	 vaalpha01 = BCAST(A[(i+1)*lda+(k+5)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb13, gvl); // sum += ALPHA*A*B

                vaalpha2 = BCAST(A[(i+2)*lda+(k+4)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb12, gvl); // sum += ALPHA*A*B

                vaalpha02 = BCAST(A[(i+2)*lda+(k+5)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb13, gvl); // sum += ALPHA*A*B


                vaalpha3 = BCAST(A[(i+3)*lda+(k+4)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb12, gvl); // sum += ALPHA*A*B

                vaalpha03 = BCAST(A[(i+3)*lda+(k+5)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb13, gvl); // sum += ALPHA*A*B

                vaalpha4 = BCAST(A[(i+4)*lda+(k+4)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb12, gvl); // sum += ALPHA*A*B

                vaalpha04 = BCAST(A[(i+4)*lda+(k+5)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb13, gvl); // sum += ALPHA*A*B

                vaalpha5 = BCAST(A[(i+5)*lda+(k+4)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb12, gvl); // sum += ALPHA*A*B

               vaalpha05 = BCAST( A[(i+5)*lda+(k+5)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb13, gvl); // sum += ALPHA*A*B

                vaalpha6 = BCAST(A[(i+6)*lda+(k+4)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb12, gvl); // sum += ALPHA*A*B

                vaalpha06 = BCAST(A[(i+6)*lda+(k+5)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb13, gvl); // sum += ALPHA*A*B

                vaalpha7 = BCAST(A[(i+7)*lda+(k+4)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb12, gvl); // sum += ALPHA*A*B

                vaalpha07 = BCAST(A[(i+7)*lda+(k+5)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb13, gvl); // sum += ALPHA*A*B


                vaalpha8 = BCAST(A[(i+8)*lda+(k+4)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb12, gvl); // sum += ALPHA*A*B

                vaalpha08 = BCAST(A[(i+8)*lda+(k+5)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb13, gvl); // sum += ALPHA*A*B


                vaalpha9= BCAST(A[(i+9)*lda+(k+4)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb12, gvl); // sum += ALPHA*A*B

                vaalpha09= BCAST(A[(i+9)*lda+(k+5)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb13, gvl); // sum += ALPHA*A*B


                vaalpha10 = BCAST(A[(i+10)*lda+(k+4)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb12, gvl); // sum += ALPHA*A*B

                vaalpha010 = BCAST(A[(i+10)*lda+(k+5)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb13, gvl); // sum += ALPHA*A*B

                vaalpha11 = BCAST(A[(i+11)*lda+(k+4)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb12, gvl); // sum += ALPHA*A*B

                vaalpha011 = BCAST(A[(i+11)*lda+(k+5)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb13, gvl); // sum += ALPHA*A*B


                vaalpha12 = BCAST(A[(i+12)*lda+(k+4)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb12, gvl); // sum += ALPHA*A*B

                vaalpha012 = BCAST(A[(i+12)*lda+(k+5)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb13, gvl); // sum += ALPHA*A*B


                vaalpha13 = BCAST(A[(i+13)*lda+(k+4)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb12, gvl); // sum += ALPHA*A*B

                vaalpha013 = BCAST(A[(i+13)*lda+(k+5)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb13, gvl); // sum += ALPHA*A*B


                vaalpha14 = BCAST(A[(i+14)*lda+(k+4)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb12, gvl); // sum += ALPHA*A*B

                vaalpha014 = BCAST(A[(i+14)*lda+(k+5)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb13, gvl); // sum += ALPHA*A*B


                vaalpha15 = BCAST(A[(i+15)*lda+(k+4)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb12, gvl); // sum += ALPHA*A*B

                vaalpha015 = BCAST(A[(i+15)*lda+(k+5)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb13, gvl); // sum += ALPHA*A*B
		  //-----
		/* unroll 8*///

                vaalpha = BCAST(A[i*lda+(k+6)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb14, gvl); // sum += ALPHA*A*B

                vaalpha0 = BCAST(A[i*lda+(k+7)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb15, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+6)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb14, gvl); // sum += ALPHA*A*B

               vaalpha01 = BCAST(A[(i+1)*lda+(k+7)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb15, gvl); // sum += ALPHA*A*B

               vaalpha2 = BCAST(A[(i+2)*lda+(k+6)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb14, gvl); // sum += ALPHA*A*B

                vaalpha02 = BCAST(A[(i+2)*lda+(k+7)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb15, gvl); // sum += ALPHA*A*B


                vaalpha3 = BCAST(A[(i+3)*lda+(k+6)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb14, gvl); // sum += ALPHA*A*B

                vaalpha03 = BCAST(A[(i+3)*lda+(k+7)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb15, gvl); // sum += ALPHA*A*B

               vaalpha4 = BCAST( A[(i+4)*lda+(k+6)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb14, gvl); // sum += ALPHA*A*B

               vaalpha04 = BCAST(A[(i+4)*lda+(k+7)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb15, gvl); // sum += ALPHA*A*B

                vaalpha5 = BCAST(A[(i+5)*lda+(k+6)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb14, gvl); // sum += ALPHA*A*B

		vaalpha05 = BCAST(A[(i+5)*lda+(k+7)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb15, gvl); // sum += ALPHA*A*B

                vaalpha6 = BCAST(A[(i+6)*lda+(k+6)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb14, gvl); // sum += ALPHA*A*B

		vaalpha06 = BCAST(A[(i+6)*lda+(k+7)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb15, gvl); // sum += ALPHA*A*B

                 vaalpha7 = BCAST(A[(i+7)*lda+(k+6)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb14, gvl); // sum += ALPHA*A*B

                  vaalpha07 = BCAST(A[(i+7)*lda+(k+7)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb15, gvl); // sum += ALPHA*A*B


                  vaalpha8 = BCAST(A[(i+8)*lda+(k+6)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb14, gvl); // sum += ALPHA*A*B

                  vaalpha08 = BCAST(A[(i+8)*lda+(k+7)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb15, gvl); // sum += ALPHA*A*B


                   vaalpha9= BCAST(A[(i+9)*lda+(k+6)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb14, gvl); // sum += ALPHA*A*B

               	 vaalpha09= BCAST(A[(i+9)*lda+(k+7)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb15, gvl); // sum += ALPHA*A*B


                 vaalpha10 = BCAST(A[(i+10)*lda+(k+6)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb14, gvl); // sum += ALPHA*A*B

                 vaalpha010 = BCAST(A[(i+10)*lda+(k+7)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb15, gvl); // sum += ALPHA*A*B

                vaalpha11 = BCAST(A[(i+11)*lda+(k+6)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb14, gvl); // sum += ALPHA*A*B

                 vaalpha011 = BCAST(A[(i+11)*lda+(k+7)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb15, gvl); // sum += ALPHA*A*B


                vaalpha12 = BCAST(A[(i+12)*lda+(k+6)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb14, gvl); // sum += ALPHA*A*B

                vaalpha012 = BCAST(A[(i+12)*lda+(k+7)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb15, gvl); // sum += ALPHA*A*B


                 vaalpha13 = BCAST(A[(i+13)*lda+(k+6)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb14, gvl); // sum += ALPHA*A*B

		vaalpha013 = BCAST( A[(i+13)*lda+(k+7)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb15, gvl); // sum += ALPHA*A*B


                   vaalpha14 = BCAST(A[(i+14)*lda+(k+6)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb14, gvl); // sum += ALPHA*A*B

                 vaalpha014 = BCAST(A[(i+14)*lda+(k+7)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb15, gvl); // sum += ALPHA*A*B


                   vaalpha15 = BCAST(A[(i+15)*lda+(k+6)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb14, gvl); // sum += ALPHA*A*B

                   vaalpha015 = BCAST(A[(i+15)*lda+(k+7)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb15, gvl); // sum += ALPHA*A*B

	}
	else
	{

               __epi_2xf32 vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha0 = BCAST(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B

		__epi_2xf32 vaalpha01 = BCAST(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha02 = BCAST(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha03 = BCAST(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha04 = BCAST(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha05 = BCAST( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha06 = BCAST(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha07 = BCAST(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha08 = BCAST(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha9= BCAST(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha09= BCAST(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha010 = BCAST(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha011 = BCAST(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha012 = BCAST(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha013 = BCAST(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha014 = BCAST(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha15 = BCAST(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha015 = BCAST(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = BCAST(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb2, gvl); // sum += ALPHA*A*B

                vaalpha0 = BCAST(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B

               vaalpha01 = BCAST(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B

               vaalpha2 = BCAST(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B

                vaalpha02 = BCAST(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B


                vaalpha3 = BCAST(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B

                vaalpha03 = BCAST(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B

               vaalpha4 = BCAST( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B

               vaalpha04 = BCAST(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B

                vaalpha5 = BCAST(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B

		vaalpha05 = BCAST(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B

                vaalpha6 = BCAST(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B

		vaalpha06 = BCAST(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B

                 vaalpha7 = BCAST(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B

                  vaalpha07 = BCAST(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B


                  vaalpha8 = BCAST(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B

                  vaalpha08 = BCAST(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B


                   vaalpha9= BCAST(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B

               	 vaalpha09= BCAST(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B


                 vaalpha10 = BCAST(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B

                 vaalpha010 = BCAST(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B

                vaalpha11 = BCAST(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B

                 vaalpha011 = BCAST(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B


                vaalpha12 = BCAST(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B

                vaalpha012 = BCAST(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B


                 vaalpha13 = BCAST(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B

		vaalpha013 = BCAST( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B


                   vaalpha14 = BCAST(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B

                 vaalpha014 = BCAST(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B


                   vaalpha15 = BCAST(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B

                   vaalpha015 = BCAST(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

		/*******/
		// unroll 6
		 vaalpha = BCAST(A[i*lda+(k+4)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb4, gvl); // sum += ALPHA*A*B

                vaalpha0 = BCAST(A[i*lda+(k+5)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb5, gvl); // sum += ALPHA*A*B

                vaalpha1 = BCAST(A[(i+1)*lda+(k+4)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb4, gvl); // sum += ALPHA*A*B

		 vaalpha01 = BCAST(A[(i+1)*lda+(k+5)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb5, gvl); // sum += ALPHA*A*B

                 vaalpha2 = BCAST(A[(i+2)*lda+(k+4)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb4, gvl); // sum += ALPHA*A*B

                vaalpha02 = BCAST(A[(i+2)*lda+(k+5)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb5, gvl); // sum += ALPHA*A*B


                vaalpha3 = BCAST(A[(i+3)*lda+(k+4)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb4, gvl); // sum += ALPHA*A*B

                vaalpha03 = BCAST(A[(i+3)*lda+(k+5)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb5, gvl); // sum += ALPHA*A*B

                vaalpha4 = BCAST(A[(i+4)*lda+(k+4)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb4, gvl); // sum += ALPHA*A*B

                vaalpha04 = BCAST(A[(i+4)*lda+(k+5)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb5, gvl); // sum += ALPHA*A*B

                vaalpha5 = BCAST(A[(i+5)*lda+(k+4)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb4, gvl); // sum += ALPHA*A*B

                vaalpha05 = BCAST( A[(i+5)*lda+(k+5)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb5, gvl); // sum += ALPHA*A*B

                vaalpha6 = BCAST(A[(i+6)*lda+(k+4)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb4, gvl); // sum += ALPHA*A*B

                vaalpha06 = BCAST(A[(i+6)*lda+(k+5)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb5, gvl); // sum += ALPHA*A*B

                vaalpha7 = BCAST(A[(i+7)*lda+(k+4)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb4, gvl); // sum += ALPHA*A*B

                vaalpha07 = BCAST(A[(i+7)*lda+(k+5)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb5, gvl); // sum += ALPHA*A*B


                vaalpha8 = BCAST(A[(i+8)*lda+(k+4)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb4, gvl); // sum += ALPHA*A*B

                vaalpha08 = BCAST(A[(i+8)*lda+(k+5)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb5, gvl); // sum += ALPHA*A*B


                vaalpha9= BCAST(A[(i+9)*lda+(k+4)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb4, gvl); // sum += ALPHA*A*B

                vaalpha09= BCAST(A[(i+9)*lda+(k+5)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb5, gvl); // sum += ALPHA*A*B


                vaalpha10 = BCAST(A[(i+10)*lda+(k+4)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb4, gvl); // sum += ALPHA*A*B

                vaalpha010 = BCAST(A[(i+10)*lda+(k+5)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb5, gvl); // sum += ALPHA*A*B

                vaalpha11 = BCAST(A[(i+11)*lda+(k+4)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb4, gvl); // sum += ALPHA*A*B

                vaalpha011 = BCAST(A[(i+11)*lda+(k+5)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb5, gvl); // sum += ALPHA*A*B


                vaalpha12 = BCAST(A[(i+12)*lda+(k+4)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb4, gvl); // sum += ALPHA*A*B

                vaalpha012 = BCAST(A[(i+12)*lda+(k+5)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb5, gvl); // sum += ALPHA*A*B


                vaalpha13 = BCAST(A[(i+13)*lda+(k+4)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb4, gvl); // sum += ALPHA*A*B

                vaalpha013 = BCAST(A[(i+13)*lda+(k+5)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb5, gvl); // sum += ALPHA*A*B


                vaalpha14 = BCAST(A[(i+14)*lda+(k+4)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb4, gvl); // sum += ALPHA*A*B

               vaalpha014 = BCAST(A[(i+14)*lda+(k+5)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb5, gvl); // sum += ALPHA*A*B


               vaalpha15 = BCAST(A[(i+15)*lda+(k+4)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb4, gvl); // sum += ALPHA*A*B

                vaalpha015 = BCAST(A[(i+15)*lda+(k+5)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb5, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 8*/

                vaalpha = BCAST(A[i*lda+(k+6)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb6, gvl); // sum += ALPHA*A*B

                vaalpha0 = BCAST(A[i*lda+(k+7)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb7, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+6)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb6, gvl); // sum += ALPHA*A*B

               vaalpha01 = BCAST(A[(i+1)*lda+(k+7)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb7, gvl); // sum += ALPHA*A*B

               vaalpha2 = BCAST(A[(i+2)*lda+(k+6)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb6, gvl); // sum += ALPHA*A*B

                vaalpha02 = BCAST(A[(i+2)*lda+(k+7)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb7, gvl); // sum += ALPHA*A*B


                vaalpha3 = BCAST(A[(i+3)*lda+(k+6)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb6, gvl); // sum += ALPHA*A*B

                vaalpha03 = BCAST(A[(i+3)*lda+(k+7)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb7, gvl); // sum += ALPHA*A*B

               vaalpha4 = BCAST( A[(i+4)*lda+(k+6)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb6, gvl); // sum += ALPHA*A*B

               vaalpha04 = BCAST(A[(i+4)*lda+(k+7)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb7, gvl); // sum += ALPHA*A*B

                vaalpha5 = BCAST(A[(i+5)*lda+(k+6)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb6, gvl); // sum += ALPHA*A*B

		vaalpha05 = BCAST(A[(i+5)*lda+(k+7)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb7, gvl); // sum += ALPHA*A*B

                vaalpha6 = BCAST(A[(i+6)*lda+(k+6)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb6, gvl); // sum += ALPHA*A*B

		vaalpha06 = BCAST(A[(i+6)*lda+(k+7)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb7, gvl); // sum += ALPHA*A*B

                 vaalpha7 = BCAST(A[(i+7)*lda+(k+6)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb6, gvl); // sum += ALPHA*A*B

                  vaalpha07 = BCAST(A[(i+7)*lda+(k+7)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb7, gvl); // sum += ALPHA*A*B


                  vaalpha8 = BCAST(A[(i+8)*lda+(k+6)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb6, gvl); // sum += ALPHA*A*B

                  vaalpha08 = BCAST(A[(i+8)*lda+(k+7)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb7, gvl); // sum += ALPHA*A*B


                   vaalpha9= BCAST(A[(i+9)*lda+(k+6)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb6, gvl); // sum += ALPHA*A*B

               	 vaalpha09= BCAST(A[(i+9)*lda+(k+7)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb7, gvl); // sum += ALPHA*A*B


                 vaalpha10 = BCAST(A[(i+10)*lda+(k+6)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb6, gvl); // sum += ALPHA*A*B

                 vaalpha010 = BCAST(A[(i+10)*lda+(k+7)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb7, gvl); // sum += ALPHA*A*B

                vaalpha11 = BCAST(A[(i+11)*lda+(k+6)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb6, gvl); // sum += ALPHA*A*B

                 vaalpha011 = BCAST(A[(i+11)*lda+(k+7)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb7, gvl); // sum += ALPHA*A*B


                vaalpha12 = BCAST(A[(i+12)*lda+(k+6)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb6, gvl); // sum += ALPHA*A*B

                vaalpha012 = BCAST(A[(i+12)*lda+(k+7)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb7, gvl); // sum += ALPHA*A*B


                 vaalpha13 = BCAST(A[(i+13)*lda+(k+6)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb6, gvl); // sum += ALPHA*A*B

		vaalpha013 = BCAST( A[(i+13)*lda+(k+7)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb7, gvl); // sum += ALPHA*A*B


                   vaalpha14 = BCAST(A[(i+14)*lda+(k+6)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb6, gvl); // sum += ALPHA*A*B

                 vaalpha014 = BCAST(A[(i+14)*lda+(k+7)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb7, gvl); // sum += ALPHA*A*B


                   vaalpha15 = BCAST(A[(i+15)*lda+(k+6)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb6, gvl); // sum += ALPHA*A*B

                   vaalpha015 = BCAST(A[(i+15)*lda+(k+7)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb7, gvl); // sum += ALPHA*A*B

	}
	flag++;
	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha = ALPHA * A[i*lda+k1];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 = ALPHA * A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
    STORE_C_UNROLL_16(0,0);
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;

     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = BCAST(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = BCAST(A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = BCAST(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}

/***** new kernel end*////



/***********************3. loop interchange with manual vectorization with ALPHA=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_noalpha_doublebuff(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7,vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;
        LOAD_C_UNROLL_16();
	// double buffer scheme implementation -start 
        int flag=0;
	for ( k = 0; k < K-3; k +=4) {
		 if (flag==0){
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);
                 vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                }
                else
                {
			if(flag & 1) 
			{
			   if(k<K-4)
			   {
                 		vb = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 		vb1 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 		vb2 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 		vb3 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                        }
                        else
			{
			    if(k<K-4)
                           {
                                vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                                vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                                vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                                vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                	}
		}
		
		//double buffer scheme implementation - end
		if(flag & 1) 
		{

               __epi_2xf32 vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb4, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha0 = BCAST(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb5, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb4, gvl); // sum += ALPHA*A*B
               
		__epi_2xf32 vaalpha01 = BCAST(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha02 = BCAST(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha03 = BCAST(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha04 = BCAST(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha05 = BCAST( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha06 = BCAST(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb5, gvl); // sum += ALPHA*A*B
               
               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha07 = BCAST(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb5, gvl); // sum += ALPHA*A*B
               

               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha08 = BCAST(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha9= BCAST(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha09= BCAST(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha010 = BCAST(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha011 = BCAST(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha012 = BCAST(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha013 = BCAST(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha014 = BCAST(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha15 = BCAST(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha015 = BCAST(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb5, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = BCAST(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb6, gvl); // sum += ALPHA*A*B
                
                vaalpha0 = BCAST(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb7, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb6, gvl); // sum += ALPHA*A*B
               
               vaalpha01 = BCAST(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb7, gvl); // sum += ALPHA*A*B
                
               vaalpha2 = BCAST(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb6, gvl); // sum += ALPHA*A*B
	
                vaalpha02 = BCAST(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb7, gvl); // sum += ALPHA*A*B
                

                vaalpha3 = BCAST(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb6, gvl); // sum += ALPHA*A*B
		
                vaalpha03 = BCAST(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb7, gvl); // sum += ALPHA*A*B
                
               vaalpha4 = BCAST( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb6, gvl); // sum += ALPHA*A*B
	
               vaalpha04 = BCAST(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb7, gvl); // sum += ALPHA*A*B
                
                vaalpha5 = BCAST(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb6, gvl); // sum += ALPHA*A*B
               
		vaalpha05 = BCAST(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb7, gvl); // sum += ALPHA*A*B
                
                vaalpha6 = BCAST(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb6, gvl); // sum += ALPHA*A*B
                 
		vaalpha06 = BCAST(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb7, gvl); // sum += ALPHA*A*B
               
                 vaalpha7 = BCAST(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb6, gvl); // sum += ALPHA*A*B
		 
                  vaalpha07 = BCAST(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb7, gvl); // sum += ALPHA*A*B
               

                  vaalpha8 = BCAST(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb6, gvl); // sum += ALPHA*A*B
		  
                  vaalpha08 = BCAST(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb7, gvl); // sum += ALPHA*A*B
                

                   vaalpha9= BCAST(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb6, gvl); // sum += ALPHA*A*B
		  
               	 vaalpha09= BCAST(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb7, gvl); // sum += ALPHA*A*B
                

                 vaalpha10 = BCAST(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb6, gvl); // sum += ALPHA*A*B
		 
                 vaalpha010 = BCAST(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb7, gvl); // sum += ALPHA*A*B
                
                vaalpha11 = BCAST(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb6, gvl); // sum += ALPHA*A*B
		 
                 vaalpha011 = BCAST(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb7, gvl); // sum += ALPHA*A*B
                

                vaalpha12 = BCAST(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb6, gvl); // sum += ALPHA*A*B
		
                vaalpha012 = BCAST(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb7, gvl); // sum += ALPHA*A*B
                

                 vaalpha13 = BCAST(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb6, gvl); // sum += ALPHA*A*B
                 
		vaalpha013 = BCAST( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb7, gvl); // sum += ALPHA*A*B
                

                   vaalpha14 = BCAST(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb6, gvl); // sum += ALPHA*A*B
		  
                 vaalpha014 = BCAST(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb7, gvl); // sum += ALPHA*A*B
                

                   vaalpha15 = BCAST(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb6, gvl); // sum += ALPHA*A*B
		   
                   vaalpha015 = BCAST(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb7, gvl); // sum += ALPHA*A*B
	
	}
	else
	{

               __epi_2xf32 vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha0 = BCAST(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               
		__epi_2xf32 vaalpha01 = BCAST(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha02 = BCAST(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha03 = BCAST(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha04 = BCAST(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha05 = BCAST( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha06 = BCAST(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha07 = BCAST(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha08 = BCAST(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha9= BCAST(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha09= BCAST(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha010 = BCAST(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha011 = BCAST(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha012 = BCAST(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha013 = BCAST(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha014 = BCAST(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha15 = BCAST(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha015 = BCAST(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = BCAST(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                
                vaalpha0 = BCAST(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
               
               vaalpha01 = BCAST(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha2 = BCAST(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
	
                vaalpha02 = BCAST(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha3 = BCAST(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha03 = BCAST(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha4 = BCAST( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
	
               vaalpha04 = BCAST(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha5 = BCAST(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
               
		vaalpha05 = BCAST(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha6 = BCAST(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha06 = BCAST(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
                 vaalpha7 = BCAST(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		 
                  vaalpha07 = BCAST(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

                  vaalpha8 = BCAST(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  
                  vaalpha08 = BCAST(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha9= BCAST(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  
               	 vaalpha09= BCAST(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha10 = BCAST(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha010 = BCAST(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha11 = BCAST(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha011 = BCAST(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha12 = BCAST(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha012 = BCAST(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha13 = BCAST(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha013 = BCAST( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha14 = BCAST(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  
                 vaalpha014 = BCAST(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha15 = BCAST(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   
                   vaalpha015 = BCAST(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

	}
	flag++;
	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha =  A[i*lda+k1];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 =  A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 =  A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 =  A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 =  A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 =  A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 =  A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 =  A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 =  A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 =  A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 =  A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 =  A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 =  A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 =  A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 =  A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 =  A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
    STORE_C_UNROLL_16(0,0);
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = BCAST(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = BCAST(A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = BCAST(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization ALPHA!=1 with double buffer****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_doublebuff(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vb4, vb5, vb6, vb7, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;
        LOAD_C_UNROLL_16();
	//
	int flag=0;
        for ( k = 0; k < K-3; k +=4) {
		// double buffer scheme implementation -start 
		  if (flag==0){
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);
                 vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                }
                else
                {
                        if(flag & 1)
                        {
                           if(k<K-4)
                           {
                                vb = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                                vb1 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                                vb2 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                                vb3 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                        }
                        else
                        {
                            if(k<K-4)
                           {
                                vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                                vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                                vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                                vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                        }
                }

		// double buffer scheme implementation - end

		if(flag & 1)
		{
                register float alpha = ALPHA * A[i*lda+k];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb4, gvl); // sum += ALPHA*A*B
                register float alpha0 = ALPHA * A[i*lda+(k+1)];
               __epi_2xf32 vaalpha0 = BCAST(alpha0, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb5, gvl); // sum += ALPHA*A*B

                register float alpha1 = ALPHA * A[(i+1)*lda+k];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb4, gvl); // sum += ALPHA*A*B
                register float alpha01 = ALPHA * A[(i+1)*lda+(k+1)];
               __epi_2xf32 vaalpha01 = BCAST(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha2 = ALPHA * A[(i+2)*lda+k];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb4, gvl); // sum += ALPHA*A*B
		register float alpha02 = ALPHA * A[(i+2)*lda+(k+1)];
               __epi_2xf32 vaalpha02 = BCAST(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha3 = ALPHA * A[(i+3)*lda+k];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb4, gvl); // sum += ALPHA*A*B
		register float alpha03 = ALPHA * A[(i+3)*lda+(k+1)];
               __epi_2xf32 vaalpha03 = BCAST(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha4 = ALPHA * A[(i+4)*lda+k];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb4, gvl); // sum += ALPHA*A*B
		register float alpha04 = ALPHA * A[(i+4)*lda+(k+1)];
               __epi_2xf32 vaalpha04 = BCAST(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha5 = ALPHA * A[(i+5)*lda+k];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb4, gvl); // sum += ALPHA*A*B
		register float alpha05 = ALPHA * A[(i+5)*lda+(k+1)];
               __epi_2xf32 vaalpha05 = BCAST(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha6 = ALPHA * A[(i+6)*lda+k];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb4, gvl); // sum += ALPHA*A*B
		register float alpha06 = ALPHA * A[(i+6)*lda+(k+1)];
               __epi_2xf32 vaalpha06 = BCAST(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb5, gvl); // sum += ALPHA*A*B
               
		 register float alpha7 = ALPHA * A[(i+7)*lda+k];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb4, gvl); // sum += ALPHA*A*B
		 register float alpha07 = ALPHA * A[(i+7)*lda+(k+1)];
               __epi_2xf32 vaalpha07 = BCAST(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb5, gvl); // sum += ALPHA*A*B
               

		 register float alpha8 = ALPHA * A[(i+8)*lda+k];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb4, gvl); // sum += ALPHA*A*B
		 register float alpha08 = ALPHA * A[(i+8)*lda+(k+1)];
               __epi_2xf32 vaalpha08 = BCAST(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha9 = ALPHA * A[(i+9)*lda+k];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb4, gvl); // sum += ALPHA*A*B
		register float alpha09 = ALPHA * A[(i+9)*lda+(k+1)];
               __epi_2xf32 vaalpha09= BCAST(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha10 = ALPHA * A[(i+10)*lda+k];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb4, gvl); // sum += ALPHA*A*B
		register float alpha010 = ALPHA * A[(i+10)*lda+(k+1)];
               __epi_2xf32 vaalpha010 = BCAST(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha11 = ALPHA * A[(i+11)*lda+k];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb4, gvl); // sum += ALPHA*A*B
		register float alpha011 = ALPHA * A[(i+11)*lda+(k+1)];
               __epi_2xf32 vaalpha011 = BCAST(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha12 = ALPHA * A[(i+12)*lda+k];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb4, gvl); // sum += ALPHA*A*B
		register float alpha012 = ALPHA * A[(i+12)*lda+(k+1)];
               __epi_2xf32 vaalpha012 = BCAST(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha13 = ALPHA * A[(i+13)*lda+k];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb4, gvl); // sum += ALPHA*A*B
		register float alpha013 = ALPHA * A[(i+13)*lda+(k+1)];
               __epi_2xf32 vaalpha013 = BCAST(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha14 = ALPHA * A[(i+14)*lda+k];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb4, gvl); // sum += ALPHA*A*B
		register float alpha014 = ALPHA * A[(i+14)*lda+(k+1)];
               __epi_2xf32 vaalpha014 = BCAST(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha15 = ALPHA * A[(i+15)*lda+k];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb4, gvl); // sum += ALPHA*A*B
		register float alpha015 = ALPHA * A[(i+15)*lda+(k+1)];
               __epi_2xf32 vaalpha015 = BCAST(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb5, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                alpha = ALPHA * A[i*lda+(k+2)];
                vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb6, gvl); // sum += ALPHA*A*B
                 alpha0 = ALPHA * A[i*lda+(k+3)];
                vaalpha0 = BCAST(alpha0, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb7, gvl); // sum += ALPHA*A*B

                alpha1 = ALPHA * A[(i+1)*lda+(k+2)];
               vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb6, gvl); // sum += ALPHA*A*B
                alpha01 = ALPHA * A[(i+1)*lda+(k+3)];
               vaalpha01 = BCAST(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb7, gvl); // sum += ALPHA*A*B
                
		 alpha2 = ALPHA * A[(i+2)*lda+(k+2)];
               vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb6, gvl); // sum += ALPHA*A*B
		alpha02 = ALPHA * A[(i+2)*lda+(k+3)];
                vaalpha02 = BCAST(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb7, gvl); // sum += ALPHA*A*B
                

		alpha3 = ALPHA * A[(i+3)*lda+(k+2)];
                vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb6, gvl); // sum += ALPHA*A*B
		alpha03 = ALPHA * A[(i+3)*lda+(k+3)];
                vaalpha03 = BCAST(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb7, gvl); // sum += ALPHA*A*B
                
		alpha4 = ALPHA * A[(i+4)*lda+(k+2)];
               vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb6, gvl); // sum += ALPHA*A*B
		alpha04 = ALPHA * A[(i+4)*lda+(k+3)];
               vaalpha04 = BCAST(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb7, gvl); // sum += ALPHA*A*B
                
		alpha5 = ALPHA * A[(i+5)*lda+(k+2)];
                vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb6, gvl); // sum += ALPHA*A*B
		 alpha05 = ALPHA * A[(i+5)*lda+(k+3)];
               vaalpha05 = BCAST(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb7, gvl); // sum += ALPHA*A*B
                
		alpha6 = ALPHA * A[(i+6)*lda+(k+2)];
                vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb6, gvl); // sum += ALPHA*A*B
		 alpha06 = ALPHA * A[(i+6)*lda+(k+3)];
                 vaalpha06 = BCAST(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb7, gvl); // sum += ALPHA*A*B
               
		 alpha7 = ALPHA * A[(i+7)*lda+(k+2)];
                 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb6, gvl); // sum += ALPHA*A*B
		  alpha07 = ALPHA * A[(i+7)*lda+(k+3)];
                  vaalpha07 = BCAST(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb7, gvl); // sum += ALPHA*A*B
               

		  alpha8 = ALPHA * A[(i+8)*lda+(k+2)];
                  vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb6, gvl); // sum += ALPHA*A*B
		  alpha08 = ALPHA * A[(i+8)*lda+(k+3)];
                  vaalpha08 = BCAST(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb7, gvl); // sum += ALPHA*A*B
                

		   alpha9 = ALPHA * A[(i+9)*lda+(k+2)];
                   vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb6, gvl); // sum += ALPHA*A*B
		  alpha09 = ALPHA * A[(i+9)*lda+(k+3)];
               	 vaalpha09= BCAST(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb7, gvl); // sum += ALPHA*A*B
                

		  alpha10 = ALPHA * A[(i+10)*lda+(k+2)];
                 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb6, gvl); // sum += ALPHA*A*B
		 alpha010 = ALPHA * A[(i+10)*lda+(k+3)];
                 vaalpha010 = BCAST(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb7, gvl); // sum += ALPHA*A*B
                
		alpha11 = ALPHA * A[(i+11)*lda+(k+2)];
                vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb6, gvl); // sum += ALPHA*A*B
		 alpha011 = ALPHA * A[(i+11)*lda+(k+3)];
                 vaalpha011 = BCAST(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb7, gvl); // sum += ALPHA*A*B
                

		 alpha12 = ALPHA * A[(i+12)*lda+(k+2)];
                vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb6, gvl); // sum += ALPHA*A*B
		 alpha012 = ALPHA * A[(i+12)*lda+(k+3)];
                vaalpha012 = BCAST(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb7, gvl); // sum += ALPHA*A*B
                

		 alpha13 = ALPHA * A[(i+13)*lda+(k+2)];
                 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb6, gvl); // sum += ALPHA*A*B
		 alpha013 = ALPHA * A[(i+13)*lda+(k+3)];
                 vaalpha013 = BCAST(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb7, gvl); // sum += ALPHA*A*B
                

		  alpha14 = ALPHA * A[(i+14)*lda+(k+2)];
                   vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb6, gvl); // sum += ALPHA*A*B
		  alpha014 = ALPHA * A[(i+14)*lda+(k+3)];
                 vaalpha014 = BCAST(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb7, gvl); // sum += ALPHA*A*B
                

		   alpha15 = ALPHA * A[(i+15)*lda+(k+2)];
                   vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb6, gvl); // sum += ALPHA*A*B
		   alpha015 = ALPHA * A[(i+15)*lda+(k+3)];
                   vaalpha015 = BCAST(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb7, gvl); // sum += ALPHA*A*B
		}
		else
		{
			
                register float alpha = ALPHA * A[i*lda+k];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha0 = ALPHA * A[i*lda+(k+1)];
               __epi_2xf32 vaalpha0 = BCAST(alpha0, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

                register float alpha1 = ALPHA * A[(i+1)*lda+k];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha01 = ALPHA * A[(i+1)*lda+(k+1)];
               __epi_2xf32 vaalpha01 = BCAST(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha2 = ALPHA * A[(i+2)*lda+k];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
		register float alpha02 = ALPHA * A[(i+2)*lda+(k+1)];
               __epi_2xf32 vaalpha02 = BCAST(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha3 = ALPHA * A[(i+3)*lda+k];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
		register float alpha03 = ALPHA * A[(i+3)*lda+(k+1)];
               __epi_2xf32 vaalpha03 = BCAST(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha4 = ALPHA * A[(i+4)*lda+k];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
		register float alpha04 = ALPHA * A[(i+4)*lda+(k+1)];
               __epi_2xf32 vaalpha04 = BCAST(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha5 = ALPHA * A[(i+5)*lda+k];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
		register float alpha05 = ALPHA * A[(i+5)*lda+(k+1)];
               __epi_2xf32 vaalpha05 = BCAST(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha6 = ALPHA * A[(i+6)*lda+k];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
		register float alpha06 = ALPHA * A[(i+6)*lda+(k+1)];
               __epi_2xf32 vaalpha06 = BCAST(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
		 register float alpha7 = ALPHA * A[(i+7)*lda+k];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
		 register float alpha07 = ALPHA * A[(i+7)*lda+(k+1)];
               __epi_2xf32 vaalpha07 = BCAST(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

		 register float alpha8 = ALPHA * A[(i+8)*lda+k];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
		 register float alpha08 = ALPHA * A[(i+8)*lda+(k+1)];
               __epi_2xf32 vaalpha08 = BCAST(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha9 = ALPHA * A[(i+9)*lda+k];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
		register float alpha09 = ALPHA * A[(i+9)*lda+(k+1)];
               __epi_2xf32 vaalpha09= BCAST(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha10 = ALPHA * A[(i+10)*lda+k];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha010 = ALPHA * A[(i+10)*lda+(k+1)];
               __epi_2xf32 vaalpha010 = BCAST(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha11 = ALPHA * A[(i+11)*lda+k];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
		register float alpha011 = ALPHA * A[(i+11)*lda+(k+1)];
               __epi_2xf32 vaalpha011 = BCAST(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha12 = ALPHA * A[(i+12)*lda+k];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
		register float alpha012 = ALPHA * A[(i+12)*lda+(k+1)];
               __epi_2xf32 vaalpha012 = BCAST(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha13 = ALPHA * A[(i+13)*lda+k];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
		register float alpha013 = ALPHA * A[(i+13)*lda+(k+1)];
               __epi_2xf32 vaalpha013 = BCAST(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha14 = ALPHA * A[(i+14)*lda+k];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
		register float alpha014 = ALPHA * A[(i+14)*lda+(k+1)];
               __epi_2xf32 vaalpha014 = BCAST(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha15 = ALPHA * A[(i+15)*lda+k];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
		register float alpha015 = ALPHA * A[(i+15)*lda+(k+1)];
               __epi_2xf32 vaalpha015 = BCAST(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                alpha = ALPHA * A[i*lda+(k+2)];
                vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                 alpha0 = ALPHA * A[i*lda+(k+3)];
                vaalpha0 = BCAST(alpha0, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

                alpha1 = ALPHA * A[(i+1)*lda+(k+2)];
               vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
                alpha01 = ALPHA * A[(i+1)*lda+(k+3)];
               vaalpha01 = BCAST(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
		 alpha2 = ALPHA * A[(i+2)*lda+(k+2)];
               vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
		alpha02 = ALPHA * A[(i+2)*lda+(k+3)];
                vaalpha02 = BCAST(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

		alpha3 = ALPHA * A[(i+3)*lda+(k+2)];
                vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		alpha03 = ALPHA * A[(i+3)*lda+(k+3)];
                vaalpha03 = BCAST(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
		alpha4 = ALPHA * A[(i+4)*lda+(k+2)];
               vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
		alpha04 = ALPHA * A[(i+4)*lda+(k+3)];
               vaalpha04 = BCAST(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
		alpha5 = ALPHA * A[(i+5)*lda+(k+2)];
                vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
		 alpha05 = ALPHA * A[(i+5)*lda+(k+3)];
               vaalpha05 = BCAST(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
		alpha6 = ALPHA * A[(i+6)*lda+(k+2)];
                vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
		 alpha06 = ALPHA * A[(i+6)*lda+(k+3)];
                 vaalpha06 = BCAST(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
		 alpha7 = ALPHA * A[(i+7)*lda+(k+2)];
                 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		  alpha07 = ALPHA * A[(i+7)*lda+(k+3)];
                  vaalpha07 = BCAST(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

		  alpha8 = ALPHA * A[(i+8)*lda+(k+2)];
                  vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  alpha08 = ALPHA * A[(i+8)*lda+(k+3)];
                  vaalpha08 = BCAST(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha9 = ALPHA * A[(i+9)*lda+(k+2)];
                   vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  alpha09 = ALPHA * A[(i+9)*lda+(k+3)];
               	 vaalpha09= BCAST(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha10 = ALPHA * A[(i+10)*lda+(k+2)];
                 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 alpha010 = ALPHA * A[(i+10)*lda+(k+3)];
                 vaalpha010 = BCAST(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
		alpha11 = ALPHA * A[(i+11)*lda+(k+2)];
                vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 alpha011 = ALPHA * A[(i+11)*lda+(k+3)];
                 vaalpha011 = BCAST(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha12 = ALPHA * A[(i+12)*lda+(k+2)];
                vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		 alpha012 = ALPHA * A[(i+12)*lda+(k+3)];
                vaalpha012 = BCAST(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha13 = ALPHA * A[(i+13)*lda+(k+2)];
                 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
		 alpha013 = ALPHA * A[(i+13)*lda+(k+3)];
                 vaalpha013 = BCAST(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha14 = ALPHA * A[(i+14)*lda+(k+2)];
                   vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  alpha014 = ALPHA * A[(i+14)*lda+(k+3)];
                 vaalpha014 = BCAST(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha15 = ALPHA * A[(i+15)*lda+(k+2)];
                   vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   alpha015 = ALPHA * A[(i+15)*lda+(k+3)];
                   vaalpha015 = BCAST(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B
		}
		flag++;
	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha = ALPHA * A[i*lda+k1];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 = ALPHA * A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
    STORE_C_UNROLL_16(0,0);
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                alpha = ALPHA * A[i*lda+k];
                if (i+1 < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if (i+2 < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if (i+3 < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vaalpha = BCAST(alpha, gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = BCAST(alpha1, gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = BCAST(alpha2, gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = BCAST(alpha3, gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization with ALPHA=1****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_noalpha(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;
        LOAD_C_UNROLL_16();
	//
        for ( k = 0; k < K-3; k +=4) {
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);


               __epi_2xf32 vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha0 = BCAST(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               
		__epi_2xf32 vaalpha01 = BCAST(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha02 = BCAST(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha03 = BCAST(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha04 = BCAST(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha05 = BCAST( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha06 = BCAST(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha07 = BCAST(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha08 = BCAST(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha9= BCAST(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha09= BCAST(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha010 = BCAST(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha011 = BCAST(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha012 = BCAST(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha013 = BCAST(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha014 = BCAST(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha15 = BCAST(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha015 = BCAST(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = BCAST(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                
                vaalpha0 = BCAST(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

               vaalpha1 = BCAST(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
               
               vaalpha01 = BCAST(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha2 = BCAST(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
	
                vaalpha02 = BCAST(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha3 = BCAST(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha03 = BCAST(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha4 = BCAST( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
	
               vaalpha04 = BCAST(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha5 = BCAST(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
               
		vaalpha05 = BCAST(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha6 = BCAST(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha06 = BCAST(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
                 vaalpha7 = BCAST(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		 
                  vaalpha07 = BCAST(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

                  vaalpha8 = BCAST(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  
                  vaalpha08 = BCAST(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha9= BCAST(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  
               	 vaalpha09= BCAST(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha10 = BCAST(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha010 = BCAST(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha11 = BCAST(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha011 = BCAST(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha12 = BCAST(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha012 = BCAST(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha13 = BCAST(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha013 = BCAST( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha14 = BCAST(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  
                 vaalpha014 = BCAST(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha15 = BCAST(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   
                   vaalpha015 = BCAST(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha =  A[i*lda+k1];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 =  A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 =  A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 =  A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 =  A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 =  A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 =  A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 =  A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 =  A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 =  A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 =  A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 =  A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 =  A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 =  A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 =  A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 =  A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
    STORE_C_UNROLL_16(0,0);
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     //float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = BCAST(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = BCAST( A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = BCAST(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization ALPHA!=1****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_3loop(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;
        LOAD_C_UNROLL_16();
	//
        for ( k = 0; k < K-3; k +=4) {
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);



                register float alpha = ALPHA * A[i*lda+k];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha0 = ALPHA * A[i*lda+(k+1)];
               __epi_2xf32 vaalpha0 = BCAST(alpha0, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

                register float alpha1 = ALPHA * A[(i+1)*lda+k];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha01 = ALPHA * A[(i+1)*lda+(k+1)];
               __epi_2xf32 vaalpha01 = BCAST(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha2 = ALPHA * A[(i+2)*lda+k];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
		register float alpha02 = ALPHA * A[(i+2)*lda+(k+1)];
               __epi_2xf32 vaalpha02 = BCAST(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha3 = ALPHA * A[(i+3)*lda+k];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
		register float alpha03 = ALPHA * A[(i+3)*lda+(k+1)];
               __epi_2xf32 vaalpha03 = BCAST(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha4 = ALPHA * A[(i+4)*lda+k];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
		register float alpha04 = ALPHA * A[(i+4)*lda+(k+1)];
               __epi_2xf32 vaalpha04 = BCAST(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha5 = ALPHA * A[(i+5)*lda+k];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
		register float alpha05 = ALPHA * A[(i+5)*lda+(k+1)];
               __epi_2xf32 vaalpha05 = BCAST(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha6 = ALPHA * A[(i+6)*lda+k];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
		register float alpha06 = ALPHA * A[(i+6)*lda+(k+1)];
               __epi_2xf32 vaalpha06 = BCAST(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
		 register float alpha7 = ALPHA * A[(i+7)*lda+k];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
		 register float alpha07 = ALPHA * A[(i+7)*lda+(k+1)];
               __epi_2xf32 vaalpha07 = BCAST(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

		 register float alpha8 = ALPHA * A[(i+8)*lda+k];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
		 register float alpha08 = ALPHA * A[(i+8)*lda+(k+1)];
               __epi_2xf32 vaalpha08 = BCAST(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha9 = ALPHA * A[(i+9)*lda+k];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
		register float alpha09 = ALPHA * A[(i+9)*lda+(k+1)];
               __epi_2xf32 vaalpha09= BCAST(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha10 = ALPHA * A[(i+10)*lda+k];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha010 = ALPHA * A[(i+10)*lda+(k+1)];
               __epi_2xf32 vaalpha010 = BCAST(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha11 = ALPHA * A[(i+11)*lda+k];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
		register float alpha011 = ALPHA * A[(i+11)*lda+(k+1)];
               __epi_2xf32 vaalpha011 = BCAST(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha12 = ALPHA * A[(i+12)*lda+k];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
		register float alpha012 = ALPHA * A[(i+12)*lda+(k+1)];
               __epi_2xf32 vaalpha012 = BCAST(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha13 = ALPHA * A[(i+13)*lda+k];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
		register float alpha013 = ALPHA * A[(i+13)*lda+(k+1)];
               __epi_2xf32 vaalpha013 = BCAST(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha14 = ALPHA * A[(i+14)*lda+k];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
		register float alpha014 = ALPHA * A[(i+14)*lda+(k+1)];
               __epi_2xf32 vaalpha014 = BCAST(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha15 = ALPHA * A[(i+15)*lda+k];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
		register float alpha015 = ALPHA * A[(i+15)*lda+(k+1)];
               __epi_2xf32 vaalpha015 = BCAST(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                alpha = ALPHA * A[i*lda+(k+2)];
                vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                 alpha0 = ALPHA * A[i*lda+(k+3)];
                vaalpha0 = BCAST(alpha0, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

                alpha1 = ALPHA * A[(i+1)*lda+(k+2)];
               vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
                alpha01 = ALPHA * A[(i+1)*lda+(k+3)];
               vaalpha01 = BCAST(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
		 alpha2 = ALPHA * A[(i+2)*lda+(k+2)];
               vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
		alpha02 = ALPHA * A[(i+2)*lda+(k+3)];
                vaalpha02 = BCAST(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

		alpha3 = ALPHA * A[(i+3)*lda+(k+2)];
                vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		alpha03 = ALPHA * A[(i+3)*lda+(k+3)];
                vaalpha03 = BCAST(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
		alpha4 = ALPHA * A[(i+4)*lda+(k+2)];
               vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
		alpha04 = ALPHA * A[(i+4)*lda+(k+3)];
               vaalpha04 = BCAST(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
		alpha5 = ALPHA * A[(i+5)*lda+(k+2)];
                vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
		 alpha05 = ALPHA * A[(i+5)*lda+(k+3)];
               vaalpha05 = BCAST(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
		alpha6 = ALPHA * A[(i+6)*lda+(k+2)];
                vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
		 alpha06 = ALPHA * A[(i+6)*lda+(k+3)];
                 vaalpha06 = BCAST(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
		 alpha7 = ALPHA * A[(i+7)*lda+(k+2)];
                 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		  alpha07 = ALPHA * A[(i+7)*lda+(k+3)];
                  vaalpha07 = BCAST(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

		  alpha8 = ALPHA * A[(i+8)*lda+(k+2)];
                  vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  alpha08 = ALPHA * A[(i+8)*lda+(k+3)];
                  vaalpha08 = BCAST(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha9 = ALPHA * A[(i+9)*lda+(k+2)];
                   vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  alpha09 = ALPHA * A[(i+9)*lda+(k+3)];
               	 vaalpha09= BCAST(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha10 = ALPHA * A[(i+10)*lda+(k+2)];
                 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 alpha010 = ALPHA * A[(i+10)*lda+(k+3)];
                 vaalpha010 = BCAST(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
		alpha11 = ALPHA * A[(i+11)*lda+(k+2)];
                vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 alpha011 = ALPHA * A[(i+11)*lda+(k+3)];
                 vaalpha011 = BCAST(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha12 = ALPHA * A[(i+12)*lda+(k+2)];
                vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		 alpha012 = ALPHA * A[(i+12)*lda+(k+3)];
                vaalpha012 = BCAST(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha13 = ALPHA * A[(i+13)*lda+(k+2)];
                 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
		 alpha013 = ALPHA * A[(i+13)*lda+(k+3)];
                 vaalpha013 = BCAST(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha14 = ALPHA * A[(i+14)*lda+(k+2)];
                   vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  alpha014 = ALPHA * A[(i+14)*lda+(k+3)];
                 vaalpha014 = BCAST(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha15 = ALPHA * A[(i+15)*lda+(k+2)];
                   vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   alpha015 = ALPHA * A[(i+15)*lda+(k+3)];
                   vaalpha015 = BCAST(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha = ALPHA * A[i*lda+k1];
               __epi_2xf32 vaalpha = BCAST(alpha, gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = BCAST(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = BCAST(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = BCAST(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = BCAST(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = BCAST(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = BCAST(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = BCAST(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = BCAST(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= BCAST(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = BCAST(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 = ALPHA * A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = BCAST(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = BCAST(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = BCAST(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = BCAST(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = BCAST(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
    STORE_C_UNROLL_16(0,0);
        }
    j += gvl;
     }}
  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                alpha = ALPHA * A[i*lda+k];
                if (i+1 < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if (i+2 < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if (i+3 < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vaalpha = BCAST(alpha, gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = BCAST(alpha1, gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = BCAST(alpha2, gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = BCAST(alpha3, gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}

