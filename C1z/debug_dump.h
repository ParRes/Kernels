#include <stdio.h>
#include <assert.h>
#include <math.h>

static inline void write_out(int iteration, int n, const double * out)
{
    char fname[100] = {0};
    snprintf(fname, sizeof(fname), "out.%d.txt", iteration);
    FILE * f = fopen(fname,"w");
    assert(f!=NULL);

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (fabs(out[i*n+j]) > 0.0) {
                fprintf(f,"%5d %5d %30.15f\n", i, j, out[i*n+j]);
            }
        }
    }
    fflush(f);

    int rc = fclose(f);
    assert(rc==0);
}
