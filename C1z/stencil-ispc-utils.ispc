export uniform int ispc_num_threads() { return programCount; }

export void initialize(uniform const int n,
                       uniform double A[],
                       uniform double B[])
{
  for (uniform int i=0;i<n; i++) {
    for (uniform int j=0;j<n;j++) {
      A[i*n+j] = (double)(i+j);
      B[i*n+j] = 0.0;
    }
  }
}

#if 0
export void add(uniform const int n,
                uniform double A[])
{
  foreach (i = 0 ... n) {
    for (uniform int j=0;j<n;j++) {
      A[i*n+j] += 1.0;
    }
  }
}
#else
export void add(uniform const int n,
                uniform double A[])
{
  for (uniform int i=0;i<n; i++) {
    for (uniform int j=0;j<n;j++) {
      A[i*n+j] += 1.0;
    }
  }
}
#endif
