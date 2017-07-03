#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void add32(const int n, __global float * inout)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ( (i<n) && (j<n) ) {
        inout[i*n+j] += 1.0f;
    }
}

__kernel void add64(const int n, __global double * inout)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ( (i<n) && (j<n) ) {
        inout[i*n+j] += 1.0;
    }
}
