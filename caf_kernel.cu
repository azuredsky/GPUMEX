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


#ifndef _CAF_KERNEL_H_
#define _CAF_KERNEL_H_


static __global__ void
cafMulKernel( cufftComplex* g_sig1, cufftComplex* g_sig2, cufftComplex* g_sigt)
{

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    cufftComplex a,b,c;
    
    a.x = (float)g_sig1[idx].x;
    a.y = (float)g_sig1[idx].y;
    
    b.x = (float)g_sig2[idx].x;
    b.y = (float)g_sig2[idx].y;

    c.x = (float)(  (float)a.x * (float)b.x - 
		    (float)a.y * (float)b.y );
    c.y = (float)(  (float)a.x * (float)b.y + 
		    (float)a.y * (float)b.x );

    g_sigt[idx] = (cufftComplex)c;

}

#endif //_CAF_KERNEL_H_



