/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
* Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    Stencil

PURPOSE: This program tests the efficiency with which a space-invariant,
         linear, symmetric filter (stencil) can be applied to a square
         grid or image.
  
USAGE:   The program takes as input the linear
         dimension of the grid, and the number of iterations on the grid

               <progname> <iterations> <grid size> 
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than standard C functions, the following functions are used in 
         this program:
         wtime()

HISTORY: - Written by Rob Van der Wijngaart, February 2009.
         - RvdW: Removed unrolling pragmas for clarity;
           added constant to array "in" at end of each iteration to force 
           refreshing of neighbor data in parallel versions; August 2013
  
**********************************************************************************/

#include "prk_util.h"

#include <cmath>

const int radius = RADIUS;

#ifdef DOUBLE
typedef double prk_float_t;
const double epsilon = 1.0e-8;
#else
typedef float prk_float_t;
/* error computed in double to avoid round-off issues in reduction */
const double epsilon = 1.0e-4;
#endif

int main(int argc, char * argv[])
{
    std::cout << "Parallel Research Kernels Version " << PRKVERSION << std::endl;
    std::cout << "Serial stencil execution on 2D grid" << std::endl;

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/

  if (argc != 3 && argc !=4){
    std::cout << "Usage: " << argv[0] << " <# iterations> <array dimension> [tilesize]" << std::endl;
    return(EXIT_FAILURE);
  }

  int iterations  = std::atoi(argv[1]); /* number of times to run the algorithm */
  if (iterations < 1){
    std::cout << "ERROR: iterations must be >= 1" << iterations << std::endl;
    exit(EXIT_FAILURE);
  }

  int n  = std::atoi(argv[2]); /* linear grid dimension */
  if (n < 1){
    std::cout << "ERROR: grid dimension must be positive: " << n << std::endl;
    exit(EXIT_FAILURE);
  }

  int tilesize = 1; /* loop nest block factor */
  if (argc == 4) {
    tilesize = std::atoi(argv[3]);
    if (tilesize>n)  tilesize=n;
    if (tilesize<=0) tilesize=1;
  }

  if (radius < 1) {
    std::cout << "ERROR: Stencil radius " << radius << " should be positive " << std::endl;
    exit(EXIT_FAILURE);
  }

  if (2*radius+1 > n) {
    std::cout << "ERROR: Stencil radius " << radius << " exceeds grid size " << n << std::endl;
    exit(EXIT_FAILURE);
  }

  float_t * in  = new float_t[n*n];
  float_t * out = new float_t[n*n];

  float_t weight[2*radius+1][2*radius+1]; /* weights of points in the stencil */

  /* fill the stencil weights to reflect a discrete divergence operator */
  for (int jj=-radius; jj<=radius; jj++) {
    for (int ii=-radius; ii<=radius; ii++) {
      weight[ii+radius][jj+radius]= (prk_float_t)0;
    }
  }

#ifdef STAR
  const int stencil_size = 4*radius+1;
  for (int ii=1; ii<=radius; ii++) {
    weight[radius][radius+ii] = weight[radius+ii][radius] = static_cast<prk_float_t>(+1)/(2*ii*radius);
    weight[radius][radius-ii] = weight[radius-ii][radius] = static_cast<prk_float_t>(-1)/(2*ii*radius);
  }
#else
  const int stencil_size = (2*radius+1)*(2*radius+1);
  for (int jj=1; jj<=radius; jj++) {
    for (int ii=-jj+1; ii<jj; ii++) {
      weight[radius+ii][radius+jj]  = static_cast<prk_float_t>(+1)/(4*jj*(2*jj-1)*radius);
      weight[radius+ii][radius-jj]  = static_cast<prk_float_t>(-1)/(4*jj*(2*jj-1)*radius);
      weight[radius+jj][radius+ii]  = static_cast<prk_float_t>(+1)/(4*jj*(2*jj-1)*radius);
      weight[radius-jj][radius+ii]  = static_cast<prk_float_t>(-1)/(4*jj*(2*jj-1)*radius);
    }
    weight[radius+jj][radius+jj]    = static_cast<prk_float_t>(+1)/(4*jj*radius);
    weight[radius-jj][radius-jj]    = static_cast<prk_float_t>(-1)/(4*jj*radius);
  }
#endif

  /* interior of grid with respect to stencil */
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);

  std::cout << "Grid size            = " << n << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;
#ifdef STAR
  std::cout << "Type of stencil      = star" << std::endl;
#else
  std::cout << "Type of stencil      = compact" << std::endl;
#endif
#ifdef DOUBLE
  std::cout << "Data type            = double precision" << std::endl;
#else
  std::cout << "Data type            = single precision" << std::endl;
#endif
  std::cout << "Compact representation of stencil loop body" << std::endl;
  if ((1<tilesize) && (tilesize<n)) {
      std::cout << "Tile size            = " << tilesize << std::endl;
  } else {
      std::cout << "Untiled" << std::endl;
  }
  std::cout << "Number of iterations = " << iterations << std::endl;

  double t0 = 0.0;

  /* initialize the input and output arrays */
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      in[i*n+j] = static_cast<prk_float_t>(i) + static_cast<prk_float_t>(j);
    }
  }
  for (int i=radius; i<n-radius; i++) {
    for (int j=radius; j<n-radius; j++) {
      out[i*n+j] = static_cast<prk_float_t>(0);
    }
  }

  for (int iter = 0; iter<=iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter==1) {
      t0 = prk::wtime();
    }

    /* Apply the stencil operator */
    if ((tilesize==1) || (tilesize==n)) {
      for (int i=radius; i<n-radius; i++) {
        for (int j=radius; j<n-radius; j++) {
          #ifdef STAR
              for (int jj=-radius; jj<=radius; jj++) {
                out[i*n+j] += weight[radius][radius+jj]*in[i*n+j+jj];
              }
              for (int ii=-radius; ii<0; ii++) {
                out[i*n+j] += weight[radius+ii][radius]*in[(i+ii)*n+j];
              }
              for (int ii=1; ii<=radius; ii++) {
                out[i*n+j] += weight[radius+ii][radius]*in[(i+ii)*n+j];
              }
          #else
              for (int ii=-radius; ii<=radius; ii++) {
                for (int jj=-radius; jj<=radius; jj++) {
                  out[i*n+j] += weight[radius+ii][radius+jj]*in[(i+ii)*n+j+jj];
                }
              }
          #endif
        }
      }
    } else {
      for (int it=radius; it<n-radius; it+=tilesize) {
        for (int jt=radius; jt<n-radius; jt+=tilesize) {
          for (int i=it; i<std::min(n-radius,it+tilesize); i++) {
            for (int j=jt; j<std::min(n-radius,jt+tilesize); j++) {
              #ifdef STAR
                for (int jj=-radius; jj<=radius; jj++) {
                  out[i*n+j] += weight[radius][radius+jj]*in[i*n+j+jj];
                }
                for (int ii=-radius; ii<0; ii++) {
                  out[i*n+j] += weight[radius+ii][radius]*in[(i+ii)*n+j];
                }
                for (int ii=1; ii<=radius; ii++) {
                  out[i*n+j] += weight[radius+ii][radius]*in[(i+ii)*n+j];
                }
              #else
                for (int ii=-radius; ii<=radius; ii++) {
                  for (int jj=-radius; jj<=radius; jj++) {
                    out[i*n+j] += weight[radius+ii][radius+jj]*in[i+ii][j+jj];
                  }
                }
              #endif
            }
          }
        }
      }
    }

    /* add constant to solution to force refresh of neighbor data, if any */
    {
        for (int i=0; i<n; i++) {
          for (int j=0; j<n; j++) {
            in[i*n+j] += static_cast<prk_float_t>(1);
          }
        }
    }

  } /* end of iterations */

  double t1 = prk::wtime();
  double stencil_time = t1 - t0;

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* compute L1 norm in parallel */
  double norm = 0.0; /* the reduction will not be accurate enough to validate
                        if single precision is used here */
  for (int i=radius; i<n-radius; i++) {
    for (int j=radius; j<n-radius; j++) {
      norm += std::fabs(out[i*n+j]);
    }
  }

  norm /= active_points;

  /* verify correctness */
  double reference_norm = 2.*(iterations+1.);
  if (std::fabs(norm-reference_norm) > epsilon) {
    std::cout << "ERROR: L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
    exit(EXIT_FAILURE);
  }
  else {
    std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
    std::cout << "L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
#endif
    size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
    double avgtime = stencil_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
