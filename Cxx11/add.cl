//#define REAL float

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define REAL double

__kernel void add(const int n, __global float * inout)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ( (i<n) && (j<n) ) {
        out[i*n+j] += (REAL)1;
    }
}
