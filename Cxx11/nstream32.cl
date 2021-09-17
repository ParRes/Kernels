__kernel void nstream32(const int length, const float scalar, __global float * A, __global float * B, __global float * C)
{
    const int i = get_global_id(0);

    if (i<length) {
        A[i] += B[i] + scalar * C[i];
    }
}
