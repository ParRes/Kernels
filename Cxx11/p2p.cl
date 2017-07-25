#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifdef __cplusplus
#define MIN(x,y) std::min(x,y)
#define MAX(x,y) std::max(x,y)
#else
#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

__kernel void p2p32(const int n, __global float * grid)
{
    const int a = get_global_id(0);
    const int b = get_global_id(1);
    const int j = a*b;
    for (int i=2; i<=2*n-2; i++) {
      // for (int j=std::max(2,i-n+2); j<=std::min(i,n); j++) {
      if ( ( j >= MAX(2,i-n+2) ) && ( j <= MIN(i,n) ) )
      {
          const int x = i-j+2-1;
          const int y = j-1;
          grid[x*n+y] = grid[(x-1)*n+  y  ]
                      + grid[  x  *n+(y-1)]
                      - grid[(x-1)*n+(y-1)];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void p2p64(const int n, __global double * grid)
{
    const int a = get_global_id(0);
    const int b = get_global_id(1);
    const int j = a*b;
    for (int i=2; i<=2*n-2; i++) {
      // for (int j=std::max(2,i-n+2); j<=std::min(i,n); j++) {
      if ( ( j >= MAX(2,i-n+2) ) && ( j <= MIN(i,n) ) )
      {
          const int x = i-j+2-1;
          const int y = j-1;
          grid[x*n+y] = grid[(x-1)*n+  y  ]
                      + grid[  x  *n+(y-1)]
                      - grid[(x-1)*n+(y-1)];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
