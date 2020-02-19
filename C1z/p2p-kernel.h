#if 1

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

#else

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

#endif
