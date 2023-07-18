///
/// Copyright (c) 2013, Intel Corporation
/// Copyright (c) 2023, NVIDIA
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
#include <mpi.h>

int main(int argc, char * argv[])
{
  const int requested = MPI_THREAD_SERIALIZED;
  int provided;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided < requested) MPI_Abort(MPI_COMM_WORLD,provided);

  int me, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (me==0) {
      printf("Parallel Research Kernels version %d\n", PRKVERSION );
      printf("C11/MPI Matrix transpose: B = A^T\n");
  }

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    if (me==0) printf("Usage: <# iterations> <matrix order> [tile size]\n");
    MPI_Finalize();
    return 1;
  }

  // number of times to do the transpose
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    if (me==0) printf("ERROR: iterations must be >= 1\n");
    MPI_Finalize();
    return 1;
  }

  // order of a the matrix
  int order = atoi(argv[2]);
  if (order <= 0) {
    if (me==0) printf("ERROR: Matrix Order must be greater than 0\n");
    MPI_Finalize();
    return 1;
  }
  else if (order % np != 0) {
    if (me==0) printf("ERROR: Matrix Order %d must be evenly divisible by np=%d\n", order, np);
    MPI_Finalize();
    return 1;
  }

  const int block_order = order / np;
  if (block_order > floor(sqrt(INT_MAX))) {
    if (me==0) printf("ERROR: block_order too large - overflow risk\n");
    MPI_Finalize();
    return 1;
  }
  const int bo2 = block_order * block_order;

  // default tile size for tiling of local transpose
  int tile_size = (argc>3) ? atoi(argv[3]) : 32;
  // a negative tile size means no tiling of the local transpose
  if (tile_size <= 0) tile_size = order;

  if (me==0) {
      printf("Number of processes   = %d\n", np);
      printf("Number of iterations  = %d\n", iterations);
      printf("Matrix order          = %d\n", order);
      printf("Tile size             = %d\n", tile_size);
  }
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  const size_t bytes = (size_t)order * (size_t)block_order * sizeof(double);
  double (* const restrict A)[block_order] = (double (*)[block_order]) prk_malloc(bytes);
  double (* const restrict B)[block_order] = (double (*)[block_order]) prk_malloc(bytes);
  double (* const restrict T)[block_order] = (double (*)[block_order]) prk_malloc(bytes/np);
  if (A == NULL || B == NULL || T == NULL) {
    printf("Error allocating space; A=%p B=%p T=%p\n",A,B,T);
    MPI_Abort(MPI_COMM_WORLD,99);
  }

  for (int i=0; i<order; i++) {
    for (int j=0; j<block_order; j++) {
      A[i][j] = (double)(me * block_order + i*order + j);
      B[i][j] = 0.0;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

// Datatypes are slower
// define only 1 of these
//#define USE_SEND_DATATYPES
//#define USE_RECV_DATATYPES
#if defined(USE_SEND_DATATYPES) || defined(USE_RECV_DATATYPES)
  MPI_Datatype stride_dt;
  //int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype * newtype)
  MPI_Type_vector(block_order, 1, block_order, MPI_DOUBLE, &stride_dt);
  int dsize;
  MPI_Type_size(MPI_DOUBLE,&dsize);
  MPI_Datatype trans_dt;
  //int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype * newtype)
  MPI_Type_hvector(block_order, 1, dsize, stride_dt, &trans_dt);
  MPI_Type_commit(&trans_dt);
#endif
#if defined(USE_SEND_DATATYPES) && defined(USE_RECV_DATATYPES)
#error You can define USE_SEND_DATATYPES or USE_RECV_DATATYPES but not both!
#endif

  double t0=0.0, t1;

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
    }

    // B += A^T
    //MPI_Alltoall(A, bo2, MPI_DOUBLE, T, bo2, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int r=0; r<np; r++) {
        const int to   = (me + r) % np;
        const int from = (me - r + np) % np;
        //printf("%d: r=%d to=%d, from=%d\n", me, r, to, from);
        MPI_Sendrecv(&A[to*block_order][0],
#ifdef USE_SEND_DATATYPES
                     1,trans_dt,
#else
                     bo2,MPI_DOUBLE,
#endif
                     to,r,T,
#ifdef USE_RECV_DATATYPES
                     1,trans_dt,
#else
                     bo2,MPI_DOUBLE,
#endif
                     from,r,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        const int lo = block_order * from;
        // B(:,lo:hi) = B(:,lo:hi) + transpose(T(:,lo:hi))
        for (int i=0; i<block_order; i++) {
          for (int j=0; j<block_order; j++) {
#if defined(USE_SEND_DATATYPES) || defined(USE_RECV_DATATYPES)
            B[lo+i][j] += T[i][j];
#else
            B[lo+i][j] += T[j][i];
#endif
          }
        }
    }
    // A += 1
    for (int i=0; i<order; i++) {
      for (int j=0; j<block_order; j++) {
        A[i][j] += 1.0;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  const double trans_time = t1 - t0;
  //if (me==0) printf("trans_time=%lf\n", trans_time);

#if defined(USE_SEND_DATATYPES) || defined(USE_RECV_DATATYPES)
  MPI_Type_free(&stride_dt);
  MPI_Type_free(&trans_dt);
#endif

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.0) * (iterations*0.5);
  double abserr = 0.0;
  for (int i=0; i<order; i++) {
    for (int j=0; j<block_order; j++) {
      const double temp = (order*(me*block_order+j)+i) * (iterations+1) + addit;
      abserr += fabs(B[i][j] - temp);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &abserr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  prk_free(A);
  prk_free(B);
  prk_free(T);

#ifdef VERBOSE
  if (me==0) printf("Sum of absolute differences: %lf\n", abserr);
#endif

  const double epsilon = 1.0e-8;
  if (abserr < epsilon) {
    if (me==0) printf("Solution validates\n");
    const double avgtime = trans_time/iterations;
    size_t total_bytes = sizeof(double);
    total_bytes *= order;
    total_bytes *= order;
    //if (me==0) printf("total_bytes=%zu\n", total_bytes);
    if (me==0) printf("Rate (MB/s): %lf Avg time (s): %lf\n", 2.0e-6 * total_bytes/avgtime, avgtime );
  } else {
    if (me==0) printf("ERROR: Aggregate squared error %lf exceeds threshold %lf\n", abserr, epsilon );
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();

  return 0;
}


