#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define reg32 1    //set this value as 1 if using double buffer scheme
#define doublebuffer 0    //set this value as 1 if using double buffer scheme
#define nodoublebuffer 0    //set this value as 1 if using double buffer scheme
#define unroll24 0  //set this value as 1 if using unroll24

#define LOAD_C_IN_VEC(index) __builtin_epi_vload_2xf32(&C[(i + index) * ldc + j], gvl)
#define STORE_C_IN_MEM(index, offset_i, offset_j, vc_var) __builtin_epi_vstore_2xf32(&C[(i+offset_i+index)*ldc + (j+offset_j)], vc_var, gvl)

#define LOAD_C_UNROLL_16()\
    vc0 = LOAD_C_IN_VEC(0); \
    vc1 = LOAD_C_IN_VEC(1); \
    vc2 = LOAD_C_IN_VEC(2); \
    vc3 = LOAD_C_IN_VEC(3); \
    vc4 = LOAD_C_IN_VEC(4); \
    vc5 = LOAD_C_IN_VEC(5); \
    vc6 = LOAD_C_IN_VEC(6); \
    vc7 = LOAD_C_IN_VEC(7); \
    vc8 = LOAD_C_IN_VEC(8); \
    vc9 = LOAD_C_IN_VEC(9); \
    vc10 = LOAD_C_IN_VEC(10); \
    vc11 = LOAD_C_IN_VEC(11); \
    vc12 = LOAD_C_IN_VEC(12); \
    vc13 = LOAD_C_IN_VEC(13); \
    vc14 = LOAD_C_IN_VEC(14); \
    vc15 = LOAD_C_IN_VEC(15);

#define STORE_C_UNROLL_16(offset_i, offset_j) \
    STORE_C_IN_MEM(0, offset_i, offset_j, vc0); \
    STORE_C_IN_MEM(1, offset_i, offset_j, vc1); \
    STORE_C_IN_MEM(2, offset_i, offset_j, vc2); \
    STORE_C_IN_MEM(3, offset_i, offset_j, vc3); \
    STORE_C_IN_MEM(4, offset_i, offset_j, vc4); \
    STORE_C_IN_MEM(5, offset_i, offset_j, vc5); \
    STORE_C_IN_MEM(6, offset_i, offset_j, vc6); \
    STORE_C_IN_MEM(7, offset_i, offset_j, vc7); \
    STORE_C_IN_MEM(8, offset_i, offset_j, vc8); \
    STORE_C_IN_MEM(9, offset_i, offset_j, vc9); \
    STORE_C_IN_MEM(10, offset_i, offset_j, vc10); \
    STORE_C_IN_MEM(11, offset_i, offset_j, vc11); \
    STORE_C_IN_MEM(12, offset_i, offset_j, vc12); \
    STORE_C_IN_MEM(13, offset_i, offset_j, vc13); \
    STORE_C_IN_MEM(14, offset_i, offset_j, vc14); \
    STORE_C_IN_MEM(15, offset_i, offset_j, vc15);

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/(float)RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_opt(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_opt_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}


#if RISCV

/***********************3. loop interchange with manual vectorization with ALPHA!=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/

void gemm_nn_unroll16(int ii, int jj, int kk, float *A, float *B, float *C, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc)
{
int i1=ii, j1=jj;
  int i=0,j=0,k=0;
  long gvl;
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i = 0; i < M-15; i += 16) {
//	__builtin_prefetch(&C[(i+i1)*ldc+(j+j1)], 0, 3);
     //           __builtin_prefetch(B, 0, 2);
       //         __builtin_prefetch(A, 0, 2);
        __epi_2xf32 vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;
        LOAD_C_UNROLL_16();

        //
//	__builtin_prefetch(B, 0, 3);
  //              __builtin_prefetch(A, 0, 3);
        for ( k = 0; k < K; k ++) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32( &B[((k+(K*(j/ldb)))*ldb)+0], gvl);
               // register float alpha =  A[i+lda*k];
               __epi_2xf32 vaalpha = BCAST(A[i+lda*k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
                //register float alpha1 =  A[(i+1)+lda*k];
               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)+lda*k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                //register float alpha2 =  A[(i+2)+lda*k];
               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)+lda*k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                //register float alpha3 =  A[(i+3)+lda*k];
               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)+lda*k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
               // register float alpha4 =  A[(i+4)+lda*k];
               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)+lda*k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
               // register float alpha5 =  A[(i+5)+lda*k];
               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)+lda*k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
               // register float alpha6 =  A[(i+6)+lda*k];
               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)+lda*k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
               // register float alpha7 =  A[(i+7)+lda*k];
               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)+lda*k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
               // register float alpha8 =  A[(i+8)+lda*k];
               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)+lda*k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
               // register float alpha9 =  A[(i+9)+lda*k];
               __epi_2xf32 vaalpha9= BCAST(A[(i+9)+lda*k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
               // register float alpha10 =  A[(i+10)+lda*k];
               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)+lda*k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
               // register float alpha11 =  A[(i+11)+lda*k];
               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)+lda*k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
               // register float alpha12 =  A[(i+12)+lda*k];
               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)+lda*k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
               // register float alpha13 =  A[(i+13)+lda*k];
               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)+lda*k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
               // register float alpha14 =  A[(i+14)+lda*k];
               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)+lda*k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
               // register float alpha15 = A[(i+15)+lda*k];
		__epi_2xf32 vaalpha15 = BCAST(A[(i+15)+lda*k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
                  //-----
        }
            STORE_C_UNROLL_16(i1,j1);
                //

        }
    j += gvl;
     }

  int i_left=i;
  //itr=0;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        vc= __builtin_epi_vload_2xf32(&C[(i+i1)*ldc+(j+j1)], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+i1+1)*ldc+(j+j1)], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+i1+2)*ldc+(j+j1)], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+i1+3)*ldc+(j+j1)], gvl);}
  for (int k = 0; k < K; k ++) {
                alpha =  A[i+lda*k];
                if (i+1 < M) {alpha1 = A[(i+1)+lda*k]; }
                if (i+2 < M) { alpha2 =  A[(i+2)+lda*k];}
                if (i+3 < M) { alpha3 =  A[(i+3)+lda*k];}
                vaalpha = BCAST(alpha, gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = BCAST(alpha1, gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = BCAST(alpha2, gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = BCAST(alpha3, gvl);} // ALPHA*A
                //vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb = __builtin_epi_vload_2xf32(&B[((k+(K*(j/ldb)))*ldb)+0], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[(i+i1)*ldc+(j+j1)], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+i1+1)*ldc+(j+j1)], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+i1+2)*ldc+(j+j1)], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+i1+3)*ldc+(j+j1)], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization with ALPHA==1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/
void gemm_nn_noalpha_unroll163loops(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
	printf("3-loops");
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     //for (i = 0; i < M-23; i += 24) {
     for (i = 0; i < M-15; i += 16) {
        __epi_2xf32 vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;
        LOAD_C_UNROLL_16();
        //
	
        for ( k = 0; k < K; k ++) {
                __epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);

               __epi_2xf32 vaalpha = BCAST(A[i*lda+k], gvl); // ALPHA*A
                  vc0 = __builtin_epi_vfmacc_2xf32(vc0, vaalpha, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha1 = BCAST(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha2 = BCAST(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha3 = BCAST(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha4 = BCAST(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha5 = BCAST(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha6 = BCAST(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha7 = BCAST(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha8 = BCAST(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha9= BCAST(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha10 = BCAST(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha11 = BCAST(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha12 = BCAST(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha13 = BCAST(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha14 = BCAST(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
		__epi_2xf32 vaalpha15 = BCAST(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
                  //-----
        
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

#endif
void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

#if RISCV

//6-loops with packA and PackB
void gemm_nn_pack2(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C,  int ldc, int BlockM, int BlockN, int BlockK, float *transposeB, float *transposeA)

{        int ii,jj,kk;
	int ld = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);//16;
	//128;
	//512;//__builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
	printf("ld my val = %d", ld);
	long gvl;
	for (jj = 0; jj < N; jj+=BlockN) {
        	int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
        	for (kk = 0; kk < K; kk+=BlockK) {
                	int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
                	int itr=0;
                	for(int j=0;j<Nc;)
                	{
				gvl=__builtin_epi_vsetvl(Nc-j, __epi_e32, __epi_m1);
                       		for(int k=0;k<Kc;k++)
                        	{
                        	//      transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                                	__epi_2xf32 tmp = __builtin_epi_vload_2xf32( &B[(k+kk)*ldb+(j+jj)], gvl);
                                	//svst1(pg, &transposeB[((k+(Kc*itr))*ld)+0], tmp);
                                	__builtin_epi_vstore_2xf32( &transposeB[((k+(Kc*(j/ld)))*ld)+0], tmp, gvl);
                                	//transposeB[k*Nc+j] = B[(k+kk)*ldb+(j+jj)];
                        	}
                       	 	itr++;
				j+=gvl;
                	}
                	for (ii = 0; ii < M; ii+=BlockM) {
                        	int Mc = ((ii+BlockM >M)?(M-ii):(BlockM)) ;

				//	gvl=__builtin_epi_vsetvl(Kc-k, __epi_e32, __epi_m1);
                                	for(int i=0;i<Mc;i++)
                                	{
                        	for(int k=0;k<Kc;k++)
                        	{
                                  //      	__epi_2xf32 tmp = __builtin_epi_vload_2xf32(&A[(i+ii)*lda+(k+kk)], gvl);
                                    //    	__builtin_epi_vstore_strided_2xf32(&transposeA[k*Mc+i], tmp, Mc*4, gvl);
                                        	transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
                                	}
			//		k+=gvl;
                        	}

                       	gemm_nn_unroll16(ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc );
                	}
                }
	}
}

#endif

void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
    {
#if RISCV



	   /*** 3-loop implementation */
	gemm_nn_noalpha_unroll163loops(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#else
	gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#endif
	
    }
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_opt_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
    {
#if RISCV


	    /*** enable below for the 6-loops packed implementations*/
	float *transposeB, *transposeA;
        int blockM = ((16 >M)?M:(16)) ;
        int blockN  =((512>N)?N:(512));
        int blockK = ((128>K)?K:(128));
        transposeB= (float *)malloc(blockM*blockN*blockK*sizeof(float));
        transposeA= (float *)malloc(blockM*blockN*blockK*sizeof(float));

        if (transposeB == NULL) {
       	 	fprintf(stderr, "Fatal: failed to allocate bytes.\n");
        	exit(0);
        }
        if(transposeA == NULL) {
        	fprintf(stderr, "Fatal: failed to allocate  bytes.\n");
       		exit(0);
        }
	//gemm_nn_original(M,N,K,ALPHA,A, lda,B, ldb, C, ldc);	
	gemm_nn_pack2(M, N, K, ALPHA,A, lda, B, ldb,C, ldc, blockM, blockN, blockK, transposeB, transposeA);
	if(transposeB != NULL)
        {
                free(transposeB);
                transposeB = NULL;
        }
        if(transposeA != NULL)
        {
                free(transposeA);
                transposeA = NULL;
        }
#else
	gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#endif
	
    }
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}


#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

