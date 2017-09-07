#include <stdio.h>
#include <assert.h>

/* This only works if you include this header after DTYPE and RADIUS are defined! */

static inline void write_weights(DTYPE weight[2*RADIUS+1][2*RADIUS+1])
{
    FILE * f = fopen("weights.txt","w");
    assert(f!=NULL);

    for (int i=0; i<2*RADIUS+1; i++) {
        for (int j=0; j<2*RADIUS+1; j++) {
            if (ABS(weight[i][j]) > (DTYPE)0) {
                fprintf(f,"%2d %2d %30.15f\n", i-RADIUS, j-RADIUS, weight[i][j]);
            }
        }
    }
    fflush(f);

    int rc = fclose(f);
    assert(rc==0);
}

static inline void write_out(int iteration, int n, const DTYPE * out)
{
    char fname[100] = {0};
    snprintf(fname, sizeof(fname), "out.%d.txt", iteration);
    FILE * f = fopen(fname,"w");
    assert(f!=NULL);

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (ABS(out[i*n+j]) > (DTYPE)0) {
                fprintf(f,"%5d %5d %30.15f\n", i, j, out[i*n+j]);
            }
        }
    }
    fflush(f);

    int rc = fclose(f);
    assert(rc==0);
}
