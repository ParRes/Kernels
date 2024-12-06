#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#if 1
#define MIN(x,y) min(x,y)
#define MAX(x,y) max(x,y)
#elif defined(__cplusplus)
#define MIN(x,y) std::min(x,y)
#define MAX(x,y) std::max(x,y)
#else
#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

__kernel void p2p32(const int n, __global float * grid)
{
    const int j = get_global_id(0);
    for (int i=2; i<=2*n-2; i++) {
      // for (int j=MAX(2,i-n+2); j<=MIN(i,n); j++) {
      if ( ( j >= MAX(2,i-n+2) ) && ( j <= MIN(i,n) ) ) {
          const int x = i-j+2-1;
          const int y = j-1;
          grid[x*n+y] = grid[(x-1)*n+  y  ]
                      + grid[  x  *n+(y-1)]
                      - grid[(x-1)*n+(y-1)];
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void p2p64(const int n, __global double * grid)
{
    const int j = get_global_id(0);
#if 0
    if (j==0) {
      for (int i=2; i<=2*n-2; i++) {
        for (int j=MAX(2,i-n+2); j<=MIN(i,n); j++) {
            //printf("i,j=%d,%d\n",i,j);
            const int x = i-j+2-1;
            const int y = j-1;
            grid[x*n+y] = grid[(x-1)*n+  y  ]
                        + grid[  x  *n+(y-1)]
                        - grid[(x-1)*n+(y-1)];
        }
      }
    }
#else
    for (int i=2; i<=2*n-2; i++) {
      // for (int j=MAX(2,i-n+2); j<=MIN(i,n); j++)
      if ( ( j >= MAX(2,i-n+2) ) && ( j <= MIN(i,n) ) ) {
          //printf("i,j=%d,%d\n",i,j);
          const int x = i-j+2-1;
          const int y = j-1;
          grid[x*n+y] = grid[(x-1)*n+  y  ]
                      + grid[  x  *n+(y-1)]
                      - grid[(x-1)*n+(y-1)];
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
#endif
}
