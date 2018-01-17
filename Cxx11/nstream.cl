#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void nstream32(const int length, const float scalar, __global float * A, __global float * B, __global float * C)
{
    const int i = get_global_id(0);

    if (i<length) {
        A[i] += B[i] + scalar * C[i];
    }
}

__kernel void nstream64(const int length, const double scalar, __global double * A, __global double * B, __global double * C)
{
    const int i = get_global_id(0);

    if (i<length) {
        A[i] += B[i] + scalar * C[i];
    }
}
