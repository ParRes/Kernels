//
// This is a NAIVE implementation that may perform badly.
//
// Examples of better implementations include:
// - https://developer.apple.com/library/content/samplecode/OpenCL_Matrix_Transpose_Example/Introduction/Intro.html
// - https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclTranspose/transpose.cl
//

//#define REAL float

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define REAL double

__kernel void transpose(const int order,
                        __global REAL * a,
                        __global REAL * b)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ((i<order) && (j<order)) {
        b[i*order+j] += a[j*order+i];
        a[j*order+i] += (REAL)1;
    }
}
