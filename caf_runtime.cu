/*
 *
 * Copyright (C) 2009 Tomas Mazanec
 *
 * Author:   Tomas Mazanec <mazanec at utia.cas.cz>
 * Created:  Fri Aug 28 10:16:54 CEST 2009
 *
 * Application of CUDA in DSP algorithms 
 *	- Cross Ambiguity Function (CAF) implementation
 *
 */


#ifndef _CAF_RUNTIME_H_
#define _CAF_RUNTIME_H_


// includes, system
#include <stdlib.h>
#include <math.h>

// use only a macros !
//#include <cutil_inline.h>

#include <cufft.h>

#include "caf_kernel.cu"

// definitions

#define MAXTHREADS 512
#define MAXSHMEMSIZE 16384

////////////////////////////////////////////////////////////////////////////////

void caf_runtime(cufftComplex *h_sig1, cufftComplex *h_sig2, cufftComplex *h_sigff,
		unsigned int N, unsigned int m, unsigned int L, unsigned int ns
		)
{
//    const unsigned int N = 131072;
//    const unsigned int m = 600;
//    const unsigned int L = 5; //loop with some data-arrays only to fit device mem

    unsigned int s1size = sizeof(cufftComplex) * N;
    unsigned int s2size = sizeof(cufftComplex) * (N+m);

    unsigned int stlen1x  = N;
//    unsigned int stlen    = N * m;
//    unsigned int stlendev = N * m/L; 
//    unsigned int stsize1x = sizeof(cufftComplex) * N;
//    unsigned int stsize   = sizeof(cufftComplex) * N * m;
    unsigned int stsizedev = sizeof(cufftComplex) * N * m/L;

    unsigned int stoutlen1x = 2*ns;
    unsigned int stoutsize1x = stoutlen1x * sizeof(cufftComplex);
    
    const unsigned int NT = MAXTHREADS;
    const unsigned int NB = (unsigned int)(
				(unsigned int)N/ (unsigned int)NT);


    printf("CUDA: N=%d m=%d L=%d ns=%d NT=%d NB=%d\n\tdevice_mem_size=%dMB for output\n", 
	    N, m, L, ns, NT, NB, stsizedev/1048576);

    cufftComplex *d_sig1, *d_sig2, *d_sigt;

    // copy all to device
    // sig1
    cutilSafeCall(cudaMalloc((void**)&d_sig1, s1size));
    cutilSafeCall(cudaMemcpy(d_sig1, h_sig1, s1size, cudaMemcpyHostToDevice));
    // sig2 ; alloc s2size = s1size + m * sizeof(...)
    cutilSafeCall(cudaMalloc((void**)&d_sig2, s2size));
    // copy s1size!
    cutilSafeCall(cudaMemcpy(d_sig2, h_sig2, s1size, cudaMemcpyHostToDevice));
    // zero rest of d_sig2 mem
    cutilSafeCall( cudaMemset((void *)(d_sig2+N), 0, m*sizeof(cufftComplex)) );

    // prepare output
    cutilSafeCall(cudaMalloc((void**)&d_sigt, stsizedev));
    
    // prepare kernel
    dim3 grid(NB, 1, 1);
    dim3 block(NT, 1, 1);

    // prepare base ptrs for loop
//    cufftComplex *h_sigt_base = h_sigt;
    cufftComplex *d_sig2_base = d_sig2;
    cufftComplex *d_sigt_base = d_sigt;

    // prepare FFT
    cufftHandle plan;
    // and base ptr
    cufftComplex *h_sigff_base = h_sigff;
    
    cufftSafeCall(cufftPlan1d(&plan, N, CUFFT_C2C, m/L) );

////////////////////////////////////////////////////////////////////////////////
// BIG LOOP
////////////////////////////////////////////////////////////////////////////////

  // prepare    
  unsigned int tau = 0;
    
  for ( unsigned int loopCounter=0; loopCounter < L; loopCounter++)
  {

////////////////////////////////////////////////////////////////////////////////
// kernel exec
////////////////////////////////////////////////////////////////////////////////

    for (unsigned int k=0; k < m/L; k++)
    {
	tau = k + loopCounter*m/L;
	d_sig2 = d_sig2_base + tau; // increment ptr to sig2 by 1
	d_sigt = d_sigt_base + k*stlen1x; // incr. ptr to sigt by 1 output

        // do the sig1 .* sig2 sliding by tau
	cafMulKernel <<< grid, block >>>(d_sig1, d_sig2, d_sigt);
    }
    d_sigt = d_sigt_base;

    // check if kernel execution generated and error // when _DEBUG also calls cudaThreadSynchronize() 
    cutilCheckMsg("Kernel execution failed");
    // wait till kernel is done
    cutilSafeCall(cudaThreadSynchronize() );

////////////////////////////////////////////////////////////////////////////////
// FFT part
////////////////////////////////////////////////////////////////////////////////

    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_sigt, (cufftComplex *)d_sigt, CUFFT_FORWARD) );
    d_sigt = d_sigt_base;
    // wait till threads are done
    cutilSafeCall(cudaThreadSynchronize() );


////////////////////////////////////////////////////////////////////////////////
// data output

/*    // all data
    h_sigff = h_sigff_base + loopCounter * stlendev;
    cutilSafeCall( cudaMemcpy( (cufftComplex *)h_sigff, (cufftComplex *)d_sigt, stsizedev, cudaMemcpyDeviceToHost) );
*/
    // only 2*ns spectral lines
    h_sigff = h_sigff_base + loopCounter * stoutlen1x * m/L;

    for (unsigned int j = 0; j < (m/L+1); j++) {
        if (j == 0) {
        cutilSafeCall( cudaMemcpy( (cufftComplex *)h_sigff, (cufftComplex *)d_sigt, stoutsize1x/2, cudaMemcpyDeviceToHost) );
	h_sigff += stoutlen1x/2;
	d_sigt += ( N - stoutlen1x/2 );
	}

	if ( (j > 0) && (j < m/L) ) {
	cutilSafeCall( cudaMemcpy( (cufftComplex *)h_sigff, (cufftComplex *)d_sigt, stoutsize1x, cudaMemcpyDeviceToHost) );
	h_sigff += stoutlen1x;
	d_sigt += N;
	}
	
	if (j == m/L) {
	cutilSafeCall( cudaMemcpy( (cufftComplex *)h_sigff, (cufftComplex *)d_sigt, stoutsize1x/2, cudaMemcpyDeviceToHost) );
	}

    }
    
    d_sigt = d_sigt_base;


////////////////////////////////////////////////////////////////////////////////
// END of all computations
////////////////////////////////////////////////////////////////////////////////

  }

////////////////////////////////////////////////////////////////////////////////
// END of BIG LOOP
////////////////////////////////////////////////////////////////////////////////

    // clean-up
    cufftSafeCall( cufftDestroy(plan) );


    h_sigff = h_sigff_base;
    d_sig2 = d_sig2_base;

    // clean-up ; only device mem
    // rest is Matlab !!

    cutilSafeCall(cudaFree(d_sig1));
    cutilSafeCall(cudaFree(d_sig2));
    cutilSafeCall(cudaFree(d_sigt));

    cudaThreadExit();
    
////////////////////////////////////////////////////////////////////////////////

}

#endif // _CAF_RUNTIME_H_

