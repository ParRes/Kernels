//
// This is a NAIVE implementation that may perform badly.
//
// Examples of better implementations include:
// - https://developer.apple.com/library/content/samplecode/OpenCL_Matrix_Transpose_Example/Introduction/Intro.html
// - https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclTranspose/transpose.cl
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void transpose32(const int order, __global float * a, __global float * b)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ((i<order) && (j<order)) {
        b[i*order+j] += a[j*order+i];
        a[j*order+i] += 1.0f;
    }
}

__kernel void transpose64(const int order, __global double * a, __global double * b)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ((i<order) && (j<order)) {
        b[i*order+j] += a[j*order+i];
        a[j*order+i] += 1.0;
    }
}
