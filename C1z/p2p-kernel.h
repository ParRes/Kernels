#if UNROLLED

static inline void sweep_tile(int startm, int endm,
                              int startn, int endn,
                              int n, double g[restrict])
{
  for (int i=startm; i<endm; i++) {
    int j;
    for (j=startn; j<endn-3; j+=4) {
      //g[i*n+j] = g[(i-1)*n+j] + g[i*n+(j-1)] - g[(i-1)*n+(j-1)];
#if 1
      // WORKS
      double c0s[4] = { g[(i-1)*n+(j+0)] , g[(i-1)*n+(j+1)] , g[(i-1)*n+(j+2)] }; // shifted
      double c0r[4] = { g[(i-1)*n+(j-1)] , g[(i-1)*n+(j+0)] , g[(i-1)*n+(j+1)] }; // regular
      double i0[4]  = { c0s[0] - c0r[0] , c0s[1] - c0r[1] ,c0s[2] - c0r[2] ,c0s[3] - c0r[3] }; // subtract
      double c1[4]  = { g[  i  *n+(j-1)] , g[  i  *n+(j+0)] , g[  i  *n+(j+1)] , g[  i  *n+(j+2)] }; // regular
      double i1[4]  = { c1[0] + i0[0] , 0 , 0 };        // add first element
      double i2[4]  = { 0 , i1[0] , 0 , 0 };            // shift right
      double i3[4]  = { 0 , i2[1] + i0[1] , 0 , 0 };    // add second element
      double i4[4]  = { 0 , 0 , i3[2] , 0 };            // shift right
      double i5[4]  = { 0 , 0 , i4[2] + i0[2] , 0 };    // add third element
      double i6[4]  = { 0 , 0 , 0 , i5[2] };            // shift right
      double i7[4]  = { 0 , 0 , 0 , i6[3] + i0[3] };    // add fourth element
      g[i*n+j+0] = i1[0];
      g[i*n+j+1] = i3[1];
      g[i*n+j+2] = i5[2];
      g[i*n+j+3] = i7[3];
#else
      // WORKS
      g[i*n+j] = g[i*n+j-1] + g[(i-1)*n+j] - g[(i-1)*n+j-1];
      g[i*n+j+1] = g[i*n+j] + g[(i-1)*n+j+1] - g[(i-1)*n+j];
      g[i*n+j+2] = g[i*n+j+1] + g[(i-1)*n+j+2] - g[(i-1)*n+j+1];
      g[i*n+j+3] = g[i*n+j+2] + g[(i-1)*n+j+3] - g[(i-1)*n+j+2];
#endif
    }
    for (int jj=j; j<endn; j++) {
      g[i*n+j] = g[i*n+j-1] + g[(i-1)*n+j] - g[(i-1)*n+j-1];
    }
  }
}

#elif SCALAR_REFS

static inline void sweep_tile(int startm, int endm,
                              int startn, int endn,
                              int n, double * restrict grid)
{
    for (int i=startm; i<endm; i++) {
        double olda = grid[  i  *n+(startn-1)];
        double oldb = grid[(i-1)*n+(startn-1)];
        for (int j=startn; j<endn; j++) {
            const double newb = grid[(i-1)*n+j];
            const double newa = newb - oldb + olda;
            grid[i*n+j] = newa;
            olda = newa;
            oldb = newb;
        }
    }
}

#else

static inline void sweep_tile(int startm, int endm,
                              int startn, int endn,
                              int n, double * restrict grid)
{
  for (int i=startm; i<endm; i++) {
    for (int j=startn; j<endn; j++) {
      grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
    }
  }
}

#endif

static inline void sweep_tile_2d(int startm, int endm,
                                 int startn, int endn,
                                 int n, double (* restrict grid)[n])
{
  for (int i=startm; i<endm; i++) {
    for (int j=startn; j<endn; j++) {
      grid[i][j] = grid[i-1][j] + grid[i][j-1] - grid[i-1][j-1];
    }
  }
}

