
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
/// NAME:    Stencil
///
/// PURPOSE: This program tests the efficiency with which a space-invariant,
///          linear, symmetric filter (stencil) can be applied to a square
///          grid or image.
///
/// USAGE:   The program takes as input the linear
///          dimension of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <grid size>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following functions are used in
///          this program:
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///          - RvdW: Removed unrolling pragmas for clarity;
///            added constant to array "in" at end of each iteration to force
///            refreshing of neighbor data in parallel versions; August 2013
///            C++11-ification by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

static inline size_t offset(size_t i, size_t j, size_t lsize)
{
    return (i+(j<<lsize));
}

/* Code below reverses bits in unsigned integer stored in a 64-bit word.
   Bit reversal is with respect to the largest integer that is going to be
   processed for the particular run of the code, to make sure the reversal
   constitutes a true permutation. Hence, the final result needs to be shifted
   to the right.
   Example: if largest integer being processed is 0x000000ff = 255 =
   0000...0011111111 (binary), then the unshifted reversal of 0x00000006 = 6 =
   0000...0000000110 (binary) would be 011000000...0000 = 3*2^61, which is
   outside the range of the original sequence 0-255. Setting shift_in_bits to
   2log(256) = 8, the final result is shifted the the right by 64-8=56 bits,
   so we get 000...0001100000 (binary) = 96, which is within the proper range */

static inline uint64_t reverse(uint64_t x, int shift_in_bits)
{
  x = ((x >> 1)  & 0x5555555555555555) | ((x << 1)  & 0xaaaaaaaaaaaaaaaa);
  x = ((x >> 2)  & 0x3333333333333333) | ((x << 2)  & 0xcccccccccccccccc);
  x = ((x >> 4)  & 0x0f0f0f0f0f0f0f0f) | ((x << 4)  & 0xf0f0f0f0f0f0f0f0);
  x = ((x >> 8)  & 0x00ff00ff00ff00ff) | ((x << 8)  & 0xff00ff00ff00ff00);
  x = ((x >> 16) & 0x0000ffff0000ffff) | ((x << 16) & 0xffff0000ffff0000);
  x = ((x >> 32) & 0x00000000ffffffff) | ((x << 32) & 0xffffffff00000000);
  return ( x >> (8*sizeof(uint64_t)-shift_in_bits) );
}

#if SCRAMBLE
  #define REVERSE(a,b)  reverse((a),(b))
#else
  #define REVERSE(a,b) (a)
#endif

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11 Sparse matrix-vector multiplication" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, lsize;
  unsigned radius, stencil_size;
  size_t size, size2, nent;
  double sparsity;
  try {
      if (argc < 4) {
        throw "Usage: <# iterations> <2log grid size> <stencil radius>]";
      }

      // number of times to run the algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // linear grid dimension
      lsize  = std::atoi(argv[2]);
      if (lsize < 1) {
        throw "ERROR: grid dimension must be positive";
      }
      //size_t lsize2 = 2*lsize;
      size = 1L<<lsize;
      size2 = size*size;

      // stencil radius
      radius = std::atoi(argv[5]);

      if (radius < 0) {
        throw "ERROR: Stencil radius must be nonnegative";
      }

      stencil_size = 4*radius+1;
      sparsity = (4.*radius+1.)/size2;
      nent = size2 * stencil_size;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << size2 << std::endl;
  std::cout << "Stencil diameter     = " << 2*radius+1 << std::endl;
  std::cout << "Sparsity             = " << sparsity << std::endl;
#if SCRAMBLE
  std::cout << "Using scrambled indexing"  << std::endl;
#else
  std::cout << "Using canonical indexing"  << std::endl;
#endif

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  prk::vector<double> matrix(nent,0.0);
  prk::vector<size_t> colIndex(nent,0);
  prk::vector<double> vector(size2,0.0);
  prk::vector<double> result(size2,0.0);

  double sparse_time(0);

  {
    for (size_t row=0; row<size2; row++) {
      size_t i = row % size;
      size_t j = row / size;
      size_t elm = row*stencil_size;
      colIndex[elm] = REVERSE(offset(i,j,lsize),lsize2);
      for (size_t r=1; r<=radius; r++, elm+=4) {
        colIndex[elm+1] = REVERSE(offset((i+r)%size,j,lsize),lsize2);
        colIndex[elm+2] = REVERSE(offset((i-r+size)%size,j,lsize),lsize2);
        colIndex[elm+3] = REVERSE(offset(i,(j+r)%size,lsize),lsize2);
        colIndex[elm+4] = REVERSE(offset(i,(j-r+size)%size,lsize),lsize2);
      }
      std::sort(&(colIndex[row*stencil_size]), &(colIndex[(row+1)*stencil_size]));
      for (size_t elm=row*stencil_size; elm<(row+1)*stencil_size; elm++) {
        matrix[elm] = 1.0/(colIndex[elm]+1.);
      }
    }

    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) sparse_time = prk::wtime();

      for (size_t row=0; row<size2; row++) {
          vector[row] += (row+1.);
      }

      for (size_t row=0; row<size2; row++) {
          double temp(0);
          for (size_t col=stencil_size*row; col<stencil_size*(row+1); col++) {
              temp += matrix[col]*vector[colIndex[col]];
          }
          result[row] += temp;
      }

    }
    sparse_time = prk::wtime() - sparse_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  double reference_sum = (0.5*nent) * (iterations+1.) * (iterations+2.);

  double vector_sum(0);
  for (size_t row=0; row<size2; row++) {
      vector_sum += result[row];
  }

  const double epsilon(1.e-8);

  if (std::fabs(vector_sum-reference_sum) > epsilon) {
    std::cout << "ERROR: Vector norm = " << vector_sum
              << " Reference vector norm = " << reference_sum << std::endl;
    return 1;
  } else {
    std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
    std::cout << "Reference sum = " << reference_sum
              << ", vector sum = " << vector_sum << std::endl;
#endif
    double avgtime = sparse_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * (2.*nent)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
