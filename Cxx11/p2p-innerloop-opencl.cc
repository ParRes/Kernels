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
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an n^2 grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <n>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///            C99-ification by Jeff Hammond, February 2016.
///            C++11-ification by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_opencl.h"

template <typename T>
void run(cl::Context context, int iterations, int n)
{
  auto precision = (sizeof(T)==8) ? 64 : 32;

  cl::Program program(context, prk::opencl::loadProgram("p2p.cl"), true);

  auto function = (precision==64) ? "p2p64" : "p2p32";

  cl_int err;
  auto kernel = cl::make_kernel<int, cl::Buffer>(program, function, &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  cl::CommandQueue queue(context);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  std::vector<T> h_grid(n*n, T(0));
  for (auto j=0; j<n; j++) {
    h_grid[0*n+j] = static_cast<double>(j);
  }
  for (auto i=0; i<n; i++) {
    h_grid[i*n+0] = static_cast<double>(i);
  }

  // copy input from host to device
  cl::Buffer d_grid = cl::Buffer(context, begin(h_grid), end(h_grid), false);

  auto pipeline_time = 0.0;

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) pipeline_time = prk::wtime();

    cl::copy(queue,begin(h_grid), end(h_grid), d_grid);
    kernel(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, d_grid);
    cl::copy(queue,d_grid, begin(h_grid), end(h_grid));
    queue.finish();
    h_grid[0*n+0] = -h_grid[(n-1)*n+(n-1)];
  }
  pipeline_time = prk::wtime() - pipeline_time;

  cl::copy(d_grid, begin(h_grid), end(h_grid));

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  const T epsilon = (sizeof(T)==8) ? 1.e-8 : 1.e-4f;

  // verify correctness, using top right value
  T corner_val = ((iterations+1)*(2*n-2));
  if ( (std::fabs(h_grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << h_grid[(n-1)*n+(n-1)]
              << " does not match verification value " << corner_val << std::endl;
  }

#ifdef VERBOSE
  std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
  std::cout << "Solution validates" << std::endl;
#endif
  auto avgtime = pipeline_time/iterations;
  std::cout << "Rate (MFlops/s): "
            << 2.0e-6 * ( (n-1.)*(n-1.) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenCL INNERLOOP pipeline execution on 2D grid" << std::endl;

  prk::opencl::listPlatforms();

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int n;
  try {
      if (argc < 3) {
        throw " <# iterations> <array dimension>";
      }

      // number of times to run the pipeline algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // grid dimensions
      n = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimensions must be positive";
      } else if ( static_cast<size_t>(n)*static_cast<size_t>(n) > INT_MAX) {
        throw "ERROR: grid dimension too large - overflow risk";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << n << ", " << n << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup OpenCL environment
  //////////////////////////////////////////////////////////////////////

  cl_int err = CL_SUCCESS;

  cl::Context cpu(CL_DEVICE_TYPE_CPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(cpu) )
  {
    const int precision = prk::opencl::precision(cpu);

    std::cout << "CPU Precision         = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(cpu, iterations, n);
    } else {
        run<float>(cpu, iterations, n);
    }
  }

  cl::Context gpu(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(gpu) )
  {
    const int precision = prk::opencl::precision(gpu);

    std::cout << "GPU Precision         = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(gpu, iterations, n);
    } else {
        run<float>(gpu, iterations, n);
    }
  }

  cl::Context acc(CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(acc) )
  {

    const int precision = prk::opencl::precision(acc);

    std::cout << "ACC Precision         = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(acc, iterations, n);
    } else {
        run<float>(acc, iterations, n);
    }
  }

  return 0;
}
