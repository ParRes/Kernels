#define RESTRICT __restrict__

#if 1

inline void sweep_tile(int startm, int endm,
                       int startn, int endn,
                       int n, double * RESTRICT grid)
{
  for (int i=startm; i<endm; i++) {
    for (int j=startn; j<endn; j++) {
      grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
    }
  }
}

inline void sweep_tile(int startm, int endm,
                       int startn, int endn,
                       int n, std::vector<double> & grid)
{
  for (auto i=startm; i<endm; i++) {
    for (auto j=startn; j<endn; j++) {
      grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
    }
  }
}

inline void sweep_tile(int startm, int endm,
                       int startn, int endn,
                       int n, prk::vector<double> & grid)
{
  for (auto i=startm; i<endm; i++) {
    for (auto j=startn; j<endn; j++) {
      grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
    }
  }
}

#else

inline void sweep_tile(int startm, int endm,
                       int startn, int endn,
                       int n, double * RESTRICT grid)
{
    for (int i=startm; i<endm; i++) {
        double olda = grid[  i  *n+(startn-1)];
        double oldb = grid[(i-1)*n+(startn-1)];
        for (int j=startn; j<endn; j++) {
            double const newb = grid[(i-1)*n+j];
            double const newa = newb - oldb + olda;
            grid[i*n+j] = newa;
            olda = newa;
            oldb = newb;
        }
    }
}

inline void sweep_tile(int startm, int endm,
                       int startn, int endn,
                       int n, std::vector<double> & grid)
{
    for (int i=startm; i<endm; i++) {
        double olda = grid[  i  *n+(startn-1)];
        double oldb = grid[(i-1)*n+(startn-1)];
        for (int j=startn; j<endn; j++) {
            double const newb = grid[(i-1)*n+j];
            double const newa = newb - oldb + olda;
            grid[i*n+j] = newa;
            olda = newa;
            oldb = newb;
        }
    }
}

inline void sweep_tile(int startm, int endm,
                       int startn, int endn,
                       int n, prk::vector<double> & grid)
{
    for (int i=startm; i<endm; i++) {
        double olda = grid[  i  *n+(startn-1)];
        double oldb = grid[(i-1)*n+(startn-1)];
        for (int j=startn; j<endn; j++) {
            double const newb = grid[(i-1)*n+j];
            double const newa = newb - oldb + olda;
            grid[i*n+j] = newa;
            olda = newa;
            oldb = newb;
        }
    }
}

#endif
