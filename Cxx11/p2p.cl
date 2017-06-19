// To enable double precision, use this:
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void p2p(int phase,
                  int j,
                  int order,
                  __global float * grid)
{
    const int i = get_global_id(0);

    if (phase==0) {
      if (1<=i && i<=j) {
        int x = i;
        int y = j-i+1;
        grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
      }
    } else if (phase==1) {
      if (1<=i && i<=j) {
        int x = n+i-j-1;
        int y = n-i;
        grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
      }
    } else /* phase==2 */ {
      grid[0*n+0] = -grid[(n-1)*n+(n-1)];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
