GPUMEX
======

clmex OPENCL to matlab,nvmex CUDA to matlab


>> clmex('oclinfo.cpp')


>> oclinfo

CL_PLATFORM_NAME = NVIDIA CUDA

CL_PLATFORM_VERSION = OpenCL 1.1 CUDA 4.2.1

1 devices found


Device #0 name = GeForce G105M

	Driver version = 320.57
	
	 CL_DEVICE_ADDRESS_BITS:	32
	 
	Global Memory (MB):	512
	
	Global Memory Cache (MB):	0
	
	Local Memory (KB):	16
	
	Max clock (MHz) :	1070
	
	Max Work Group Size:	512
	
	Number of parallel compute cores:	2
>> 


>> nvmex('nv_freemem.cu')

>> nv_freemem

Device Info: 1 GPUs found in system.

After all allocation(0):     free(MB) 330     total(MB) 512 
>> 




Boost.Compute is a GPU/parallel-computing library for C++ based on OpenCL.
https://github.com/azuredsky/compute
