///
/// Copyright (c) 2013, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///          C11-ification by Jeff Hammond, June 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

typedef struct {
    int starti;
    int endi;
    int startj;
    int endj;
    int tilesize;
    int order;
    double * restrict A;
    double * restrict B;
} args_s;

#if defined(HAVE_C11_THREADS)
int transpose_tile(void * pa)
#elif defined(HAVE_PTHREADS)
void * transpose_tile(void * pa)
#endif
{
  args_s * a = (args_s*)pa;

  const int starti    = a->starti;
  const int endi      = a->endi;
  const int startj    = a->startj;
  const int endj      = a->endj;
  const int tilesize  = a->tilesize;
  const int order     = a->order;
  double * restrict A = a->A;
  double * restrict B = a->B;

#if 0
  for (int i=starti; i<endi; i++) {
    for (int j=startj; j<endj; j++) {
      B[i*order+j] += A[j*order+i];
      A[j*order+i] += 1.0;
    }
  }
#else
  for (int it=starti; it<endi; it+=tilesize) {
    for (int jt=startj; jt<endj; jt+=tilesize) {
      for (int i=it; i<MIN(endi,it+tilesize); i++) {
        for (int j=jt; j<MIN(endj,jt+tilesize); j++) {
          B[i*order+j] += A[j*order+i];
          A[j*order+i] += 1.0;
        }
      }
    }
  }
#endif

#if defined(HAVE_C11_THREADS)
  thrd_exit(0);
  return 0;
#elif defined(HAVE_PTHREADS)
  pthread_exit(NULL);
  return NULL;
#endif
}

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION );
#ifdef HAVE_C11_THREADS
  printf("C11 Threads Matrix transpose: B = A^T\n");
#else
  printf("C99/Pthreads Matrix transpose: B = A^T\n");
#endif

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 4) {
    printf("Usage: <# iterations> <matrix order> <block size> [tile size]\n");
    return 1;
  }

  // number of times to do the transpose
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // order of a the matrix
  int order = atoi(argv[2]);
  if (order <= 0) {
    printf("ERROR: Matrix Order must be greater than 0\n");
    return 1;
  }

  // default tile size for tiling of local transpose
  int block_size = (argc>3) ? atoi(argv[3]) : 256;
  // a negative tile size means no tiling of the local transpose
  if (block_size <= 0) block_size = order;

  // default tile size for tiling of local transpose
  int tile_size = (argc>4) ? atoi(argv[4]) : 32;
  // a negative tile size means no tiling of the local transpose
  if (tile_size <= 0) tile_size = block_size;

  int num_threads = order/block_size;
  if (order % block_size) num_threads++;
  num_threads *= num_threads;

  printf("Number of threads     = %d\n", num_threads);
  printf("Number of iterations  = %d\n", iterations);
  printf("Matrix order          = %d\n", order);
  printf("Block size            = %d\n", block_size);
  printf("Tile size             = %d\n", tile_size);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

#if defined(HAVE_C11_THREADS)
  thrd_t * pool = malloc(num_threads * sizeof(thrd_t));
#elif defined(HAVE_PTHREADS)
  pthread_t * pool = malloc(num_threads * sizeof(pthread_t));
#endif
  args_s * args = malloc(num_threads * sizeof(args_s));

  double trans_time = 0.0;

  size_t bytes = order*order*sizeof(double);
  double * restrict A = prk_malloc(bytes);
  double * restrict B = prk_malloc(bytes);

  {
    for (int i=0;i<order; i++) {
      for (int j=0;j<order;j++) {
        A[i*order+j] = (double)(i*order+j);
        B[i*order+j] = 0.0;
      }
    }

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) trans_time = prk_wtime();

      int tid = 0;
      for (int ib=0; ib<order; ib+=block_size) {
        for (int jb=0; jb<order; jb+=block_size) {
          args[tid].starti   = ib;
          args[tid].endi     = MIN(order,ib+block_size);
          args[tid].startj   = jb;
          args[tid].endj     = MIN(order,jb+block_size);
          args[tid].tilesize = tile_size;
          args[tid].order    = order;
          args[tid].A        = A;
          args[tid].B        = B;
#if defined(HAVE_C11_THREADS)
          int rc = thrd_create(&pool[tid], transpose_tile, &args[tid]);
          assert(rc==thrd_success);
#elif defined(HAVE_PTHREADS)
          int rc = pthread_create(&pool[tid], NULL, transpose_tile, &args[tid]);
# ifdef VERBOSE
          if (rc) printf("pthread_create = %d (EINVAL=%d, EAGAIN=%d)\n", rc, EINVAL, EAGAIN);
# endif
          assert(rc==0);
#endif
          tid++;
        }
      }
      for (int t=0; t<num_threads; t++) {
#if defined(HAVE_C11_THREADS)
        int rc = thrd_join(pool[t],NULL);
        assert(rc==thrd_success);
#elif defined(HAVE_PTHREADS)
        int rc = pthread_join(pool[t],NULL);
# ifdef VERBOSE
        if (rc) printf("pthread_join = %d (EINVAL=%d, ESRCH=%d)\n", rc, EINVAL, ESRCH);
# endif
#endif
        assert(rc==0);
      }
    }
    trans_time = prk_wtime() - trans_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr = 0.0;
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ij = i*order+j;
      const size_t ji = j*order+i;
      const double reference = (double)(ij)*(1.+iterations)+addit;
      abserr += fabs(B[ji] - reference);
    }
  }

  prk_free(A);
  prk_free(B);

  free(pool);

#ifdef VERBOSE
  printf("Sum of absolute differences: %lf\n", abserr);
#endif

  const double epsilon = 1.0e-8;
  if (abserr < epsilon) {
    printf("Solution validates\n");
    const double avgtime = trans_time/iterations;
    printf("Rate (MB/s): %lf Avg time (s): %lf\n", 2.0e-6 * bytes/avgtime, avgtime );
  } else {
    printf("ERROR: Aggregate squared error %lf exceeds threshold %lf\n", abserr, epsilon );
    return 1;
  }

  return 0;
}


