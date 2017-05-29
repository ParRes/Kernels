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

NAME:    Pipeline

PURPOSE: This program tests the efficiency with which point-to-point
         synchronization can be carried out. It does so by executing 
         a pipelined algorithm on an m*n grid. The first array dimension
         is distributed among the threads (stripwise decomposition).
  
USAGE:   The program takes as input the
         dimensions of the grid, and the number of iterations on the grid

               <progname> <iterations> <m> <n>
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than standard C functions, the following 
         functions are used in this program:

         wtime()

HISTORY: - Written by Rob Van der Wijngaart, February 2009.
           C99-ification by Jeff Hammond, February 2016.
*******************************************************************/

#include "prk_util.h"

/* error tolerance */
const double epsilon = 1.e-8;

int main(int argc, char* argv[])
{
  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/

  std::cout << "Parallel Research Kernels version" << PRKVERSION << std::endl;
  std::cout << "Serial pipeline execution on 2D grid" << std::endl;

  if (argc != 4){
    std::cout << "Usage: " << argv[0] << " <# iterations> <first array dimension> <second array dimension>" << std::endl;
    return(EXIT_FAILURE);
  }

  /* number of times to run the pipeline algorithm */
  int iterations  = std::atoi(argv[1]);
  if (iterations < 1){
    std::cout << "ERROR: iterations must be >= 1 : " << iterations << std::endl;
    exit(EXIT_FAILURE);
  }

  /* grid dimensions */
  int m = atol(argv[2]);
  int n = atol(argv[3]);

  if (m < 1 || n < 1) {
    std::cout << "ERROR: grid dimensions must be positive: " << m <<  n << std::endl;
    exit(EXIT_FAILURE);
  }

  /* working set */
  double * vector = new double[m*n];

  std::cout << "Grid sizes                = " << m << ", " << n << std::endl;
  std::cout << "Number of iterations      = " << iterations << std::endl;

  /* clear the array */
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      vector[i*n+j] = 0.0;
    }
  }
  /* set boundary values (bottom and left side of grid) */
  for (int j=0; j<n; j++) {
    vector[0*n+j] = static_cast<double>(j);
  }
  for (int i=0; i<m; i++) {
    vector[i*n+0] = static_cast<double>(i);
  }

  double pipeline_time = 0.0; /* silence compiler warning */

  for (int iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration */
    if (iter == 1) pipeline_time = prk::wtime();

    for (int i=1; i<m; i++) {
      for (int j=1; j<n; j++) {
        vector[i*n+j] = vector[(i-1)*n+j] + vector[i*n+(j-1)] - vector[(i-1)*n+(j-1)];
      }
    }

    /* copy top right corner value to bottom left corner to create dependency; we
       need a barrier to make sure the latest value is used. This also guarantees
       that the flags for the next iteration (if any) are not getting clobbered  */
    vector[0*n+0] = -vector[(m-1)*n+(n-1)];
  }

  pipeline_time = prk::wtime() - pipeline_time;

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* verify correctness, using top right value;                                  */
  double corner_val = (double)((iterations+1)*(n+m-2));
  if ( (std::fabs(vector[(m-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << vector[(m-1)*n+(n-1)]
              << " does not match verification value" << corner_val << std::endl;
    exit(EXIT_FAILURE);
  }

  delete [] vector;

#ifdef VERBOSE
  std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
  std::cout << "Solution validates" << std::endl;
#endif
  double avgtime = pipeline_time/iterations;
  std::cout << "Rate (MFlops/s): "
            << 1.0e-6 * 2. * ( static_cast<size_t>(m-1)*static_cast<size_t>(n-1) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;

  return 0;
}
