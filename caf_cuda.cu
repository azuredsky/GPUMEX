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



/* ************************************************************************ */
/* ************************************************************************ */


#include <stdlib.h>

#include "mex.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

#include "caf_runtime.cu"
#include "caf_kernel.cu"

/* ************************************************************************ */


/* ************************************************************************ */
/* ************************************************************************ */



/* ************************************************************************ */
/* ************************************************************************ */

void pack_r2c(cufftComplex *output_float, 
	      double *input_re, 
              int Ntot)
{
    int i;
    for (i = 0; i < Ntot; i++) 
    {
	output_float[i].x = input_re[i];
	output_float[i].y = 0.0f;
    }
}

/* *** */
void pack_r2c_sp(cufftComplex  *output_float,
              float *input_re,
              int Ntot)
{
    int i;
    for (i = 0; i < Ntot; i++)
    {
	output_float[i].x = input_re[i];
	output_float[i].y = 0.0f;
    }
}

/* *** */
void pack_c2c(cufftComplex *output_float, 
              double *input_re, 
              double *input_im, 
              int Ntot)
{
    int i;
    for (i = 0; i < Ntot; i++) 
    {
	output_float[i].x = input_re[i];
	output_float[i].y = input_im[i];
    }
}

/* *** */
void pack_c2c_sp(cufftComplex  *output_float,
              float *input_re,
              float *input_im,
              int Ntot)
{
    int i;
    for (i = 0; i < Ntot; i++)
    {
	output_float[i].x = input_re[i];
	output_float[i].y = input_im[i];
    }
}

/* *** */
void unpack_c2c(cufftComplex  *input_float, 
                double *output_re, 
                double *output_im,  
                int N, int tau)
{

    int i, t, imex, ic;
    
    for (i = 0; i < N; i++)
    {
        for (t = 0; t < tau; t++)
        {
	    imex = tau*i+t;
	    ic = i+t*N;
	    
	    output_re[imex] = input_float[ic].x;
	    output_im[imex] = input_float[ic].y;
        }
    }
}

/* *** */
void unpack_c2c_sp(cufftComplex  *input_float,
                float *output_re,
                float *output_im,
                int N, int tau)
{

    int i, t, imex, ic;
    
    for (i = 0; i < N; i++)
    {
        for (t = 0; t < tau; t++)
        {
	    imex = tau*i+t;
	    ic = i+t*N;
	    
	    output_re[imex] = input_float[ic].x;
	    output_im[imex] = input_float[ic].y;
        }
    }
}

/* ************************************************************************ */
/* ************************************************************************ */

void 
mexFunction(int nlhs, mxArray *plhs[], 
	    int nrhs, const mxArray *prhs[])
/*
    nlhs - number of expected mxArrays
    nrhs - number of inputs
    plhs - array of pointers to expected outputs
    prhs - array of pointers to input data
*/
{
    int M,N;
    int orM, orN;

//    int i,j;
    
    mxClassID dataClass = mxDOUBLE_CLASS;

    cufftComplex *s1, *s2, *sf;
    
    unsigned int tau_max = 1;
    unsigned int L = 1;
    unsigned int ns = 300;
    
    mexPrintf("Mex-CUDA Cross Ambiguity Function.\n");
    
    if (nrhs < 2){ mexErrMsgTxt("Need two inputs at least.\n"); }
    if (nrhs > 5){ mexErrMsgTxt("Too many input arguments.\n"); }
    
    // rows
    orM = (int)mxGetM(prhs[0]);
    M   = (int)mxGetM(prhs[1]);
    // cols
    orN = (int)mxGetN(prhs[0]);
    N   = (int)mxGetN(prhs[1]);

    if ((orM != M) || (orN !=N)) { mexErrMsgTxt("Inputs must have same dimensions.\n"); }
    if (( M>=N ) && (N != 1)) { mexErrMsgTxt("Inputs must be vectors.\n"); }
    if (( M<=N ) && (M != 1)) { mexErrMsgTxt("Inputs must be vectors.\n"); }
    
    if (( mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) && 
	( mxGetClassID(prhs[1]) == mxDOUBLE_CLASS)) {
	    mexPrintf("MEX: inputs are double precision\n");
	    dataClass = mxDOUBLE_CLASS;
    }
    else {
	if (( mxGetClassID(prhs[0]) == mxSINGLE_CLASS) && 
	    ( mxGetClassID(prhs[1]) == mxSINGLE_CLASS)) {
	    mexPrintf("MEX: inputs are single precision\n");
	    dataClass = mxSINGLE_CLASS;
	}
	else{mexErrMsgTxt("Inputs must be same precision.\n");}
    }
    
    // 3rd argument
    // get desired tau_max
    if (nrhs >= 3)
    {
	if ((mxGetN(prhs[2]) == 1) && (mxGetM(prhs[2]) == 1))
	{
	    tau_max = (unsigned int)mxGetScalar(prhs[2]);
	    mexPrintf("MEX: tau:%d\n", tau_max);
	}
	else {mexErrMsgTxt("Scalar expected as the 3rd argument\n");}
    }

    // 4th argument
    // get max number of CAF enumerations per batch
    if (nrhs >= 4)
    {
	if ((mxGetN(prhs[3]) == 1) && (mxGetM(prhs[3]) == 1))
	{
	    L = (unsigned int)mxGetScalar(prhs[3]);
	    mexPrintf("MEX: L:%d\n", L);
	}
	else {mexErrMsgTxt("Scalar expected as the 4th argument\n");}
    }
    // 5th argument
    // get desired numer of specral coefficients
    if (nrhs == 5)
    {
	if ((mxGetN(prhs[4]) == 1) && (mxGetM(prhs[4]) == 1))
	{
	    ns = (unsigned int)mxGetScalar(prhs[4]);
	    mexPrintf("MEX: NS:%d\n", ns);
	}
	else {mexErrMsgTxt("Scalar expected as the 5th argument\n");}
    }

    // Get input vectors
    // alloc
    s1  = (cufftComplex*) mxMalloc(sizeof(cufftComplex)*N*M);
    s2  = (cufftComplex*) mxMalloc(sizeof(cufftComplex)*N*M);


    // packing data
    if ( dataClass == mxDOUBLE_CLASS) 
    {
	double *s1r, *s1i, *s2r, *s2i;
	
	s1r = (double *) mxGetData(prhs[0]);
	s2r = (double *) mxGetData(prhs[1]);
	

	if(mxIsComplex(prhs[0]))
	{
	    s1i = (double *) mxGetImagData(prhs[0]);
	    pack_c2c(s1, s1r, s1i, N*M);
	}
	else 
	{
	    pack_r2c(s1, s1r, N*M);
	}
	
	if(mxIsComplex(prhs[1]))
	{
	    s2i = (double *) mxGetImagData(prhs[1]);
	    pack_c2c(s2, s2r, s2i, N*M);
	}
	else 
	{
	    pack_r2c(s2, s2r, N*M);
	}

    }
    // packing data - mxSINGLE_CLASS
    else 
    {
	float *s1r, *s1i, *s2r, *s2i;
	
	s1r = (float *) mxGetData(prhs[0]);
	s2r = (float *) mxGetData(prhs[1]);
	

	if(mxIsComplex(prhs[0]))
	{
	    s1i = (float *) mxGetImagData(prhs[0]);
	    pack_c2c_sp(s1, s1r, s1i, N*M);
	}
	else 
	{
	    pack_r2c_sp(s1, s1r, N*M);
	}
	
	if(mxIsComplex(prhs[1]))
	{
	    s2i = (float *) mxGetImagData(prhs[1]);
	    pack_c2c_sp(s2, s2r, s2i, N*M);
	}
	else 
	{
	    pack_r2c_sp(s2, s2r, N*M);
	}

    }
    // end of packing data

/* ************************************************************************ */

    sf  = (cufftComplex*) mxMalloc(sizeof(cufftComplex)*2*ns*tau_max);
    mexPrintf("MEX: host_mem_size=%dkB\n", sizeof(cufftComplex)*2*ns*tau_max/1024 );
    
/* ************************************************************************ */

/* ************************************************************************ */


    if (M==1)
    {
//	mexPrintf("MEX: N=%d tau=%d L=%d\n", N, tau_max, L, ns);
	caf_runtime(s1, s2, sf, (unsigned int)N, tau_max, L, ns);
    }
    else
    {
//    	mexPrintf("MEX: N=%d tau=%d L=%d\n", N, tau_max, L, ns);
	caf_runtime(s1, s2, sf, (unsigned int)M, tau_max, L, ns);
    }

/* ************************************************************************ */

    if (dataClass == mxDOUBLE_CLASS)
    {
	double *sfr, *sfi;

        plhs[0] = mxCreateNumericMatrix(tau_max, 2*ns, dataClass, mxCOMPLEX);
        sfr = mxGetPr(plhs[0]);
        sfi = mxGetPi(plhs[0]);
        
        unpack_c2c(sf, sfr, sfi, 2*ns, (int)tau_max);
    }

    if (dataClass == mxSINGLE_CLASS)
    {
	float *sfr, *sfi;
	
        plhs[0] = mxCreateNumericMatrix(tau_max, 2*ns, dataClass, mxCOMPLEX);
	
        sfr = (float *)mxGetPr(plhs[0]);
        sfi = (float *)mxGetPi(plhs[0]);
	
        unpack_c2c_sp(sf, sfr, sfi, 2*ns, (int)tau_max);
    }


    mxFree(sf);

    mxFree(s1);
    mxFree(s2);
    
    return;
}


/* ************************************************************************ */
/* ************************************************************************ */
