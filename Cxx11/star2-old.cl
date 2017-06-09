// To enable double precision, use this:
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// ./generate-opencl-stencil.py  star 2
// ('Type of stencil      = ', 'star')
// ('Radius of stencil    = ', 2)
// (-2, +0, -0.125)
// (-1, +0, -0.250)
// (+0, -2, -0.125)
// (+0, -1, -0.250)
// (+0, +1,  0.250)
// (+0, +2,  0.125)
// (+1, +0,  0.250)
// (+2, +0,  0.125)

__kernel void star2(const int n, __global const float * in, __global float * out)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ( (2 <= i) && (i < n-2) && (2 <= j) && (j < n-2) ) {
        out[i*n+j] += in[(i-2)*n+(j+0)] * -0.125f
                    + in[(i+0)*n+(j-2)] * -0.125f
                    + in[(i-1)*n+(j+0)] * -0.250f
                    + in[(i+0)*n+(j-1)] * -0.250f
                    + in[(i+0)*n+(j+1)] *  0.250f
                    + in[(i+1)*n+(j+0)] *  0.250f
                    + in[(i+0)*n+(j+2)] *  0.125f
                    + in[(i+2)*n+(j+0)] *  0.125f;
    }
}
