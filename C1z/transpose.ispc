#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

export uniform int ispc_num_threads() { return programCount; }

export void initialize(uniform const int order,
                       uniform double A[],
                       uniform double B[])
{
  for (uniform int i=0;i<order; i++) {
    for (uniform int j=0;j<order;j++) {
      A[i*order+j] = (double)(i*order+j);
      B[i*order+j] = 0.0;
    }
  }
}

#if 0
export void transpose(uniform const int order,
                      uniform double A[],
                      uniform double B[])
{
  foreach (i = 0 ... order) {
    for (uniform int j=0;j<order;j++) {
      B[i*order+j] += A[j*order+i];
      A[j*order+i] += 1.0;
    }
  }
}
#else
export void transpose(uniform const int order,
                      uniform double A[],
                      uniform double B[])
{
  for (uniform int i=0;i<order; i++) {
    for (uniform int j=0;j<order;j++) {
      B[i*order+j] += A[j*order+i];
      A[j*order+i] += 1.0;
    }
  }
}
#endif

#if 1
export void transpose_tiled(uniform const int order,
                            uniform double A[],
                            uniform double B[],
                            uniform const int tile_size)
{
  foreach_tiled(i = 0 ... order, j = 0 ... order) {
    B[i*order+j] += A[j*order+i];
    A[j*order+i] += 1.0;
  }
}
#else
export void transpose_tiled(uniform const int order,
                            uniform double A[],
                            uniform double B[],
                            uniform const int tile_size)
{
  for (uniform int it=0; it<order; it+=tile_size) {
    for (uniform int jt=0; jt<order; jt+=tile_size) {
      for (uniform int i=it; i<MIN(order,it+tile_size); i++) {
        for (uniform int j=jt; j<MIN(order,jt+tile_size); j++) {
          B[i*order+j] += A[j*order+i];
          A[j*order+i] += 1.0;
        }
      }
    }
  }
}
#endif
