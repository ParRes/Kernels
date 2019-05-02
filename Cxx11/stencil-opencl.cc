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
#include "prk_opencl.h"

template <typename T>
void run(cl::Context context, int iterations, int n, int radius, bool star)
{
  auto precision = (sizeof(T)==8) ? 64 : 32;

  std::string funcname1, filename1;
  funcname1.reserve(255);
  funcname1 += ( star ? "star" : "grid" );
  funcname1 += std::to_string(radius);
  filename1 = funcname1 + ( ".cl" );
  funcname1 += "_" + std::to_string(precision);
  auto funcname2 = (precision==64) ? "add64" : "add32";
  auto filename2 = "add.cl";

  std::string source = prk::opencl::loadProgram(filename1);
  if ( source==std::string("FAIL") ) {
      std::cerr << "OpenCL kernel source file (" << filename1 << ") not found. "
                << "Generating using Python script" << std::endl;
      std::string command("./generate-opencl-stencil.py ");
      command += ( star ? "star " : "grid " );
      command += std::to_string(radius);
      int rc = std::system( command.c_str() );
      if (rc != 0) {
          std::cerr << command.c_str() << " returned " << rc << std::endl;
      }
  }
  source = prk::opencl::loadProgram(filename1);
  cl::Program program1(context, source, true);
  cl::Program program2(context, prk::opencl::loadProgram(filename2), true);

  cl_int err;
  auto kernel1 = cl::make_kernel<int, cl::Buffer, cl::Buffer>(program1, funcname1, &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program1.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }
  auto kernel2 = cl::make_kernel<int, cl::Buffer>(program2, funcname2, &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program2.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  cl::CommandQueue queue(context);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  std::vector<T> h_in(n*n,  T(0));
  std::vector<T> h_out(n*n, T(0));

  auto stencil_time = 0.0;

  // initialize the input array
  for (auto i=0; i<n; i++) {
    for (auto j=0; j<n; j++) {
      h_in[i*n+j] = static_cast<T>(i+j);
    }
  }

  // copy input from host to device
  cl::Buffer d_in = cl::Buffer(context, begin(h_in), end(h_in), true);
  cl::Buffer d_out = cl::Buffer(context, begin(h_out), end(h_out), false);

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) stencil_time = prk::wtime();

    // Apply the stencil operator
    kernel1(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, d_in, d_out);
    // Add constant to solution to force refresh of neighbor data, if any
    kernel2(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, d_in);
    queue.finish();
  }
  stencil_time = prk::wtime() - stencil_time;

  // copy output back to host
  cl::copy(queue, d_out, begin(h_out), end(h_out));

#ifdef VERBOSE
  // copy input back to host - debug only
  cl::copy(queue, d_in, begin(h_in), end(h_in));
#endif

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);

  // compute L1 norm in parallel
  double norm = 0.0;
  for (auto i=radius; i<n-radius; i++) {
    for (auto j=radius; j<n-radius; j++) {
      norm += std::fabs(static_cast<double>(h_out[i*n+j]));
    }
  }
  norm /= active_points;

  // verify correctness
  const double epsilon = (sizeof(T)==8) ? 1.0e-8 : 1.0e-4;
  double reference_norm = 2*(iterations+1);
  if (std::fabs(norm-reference_norm) > epsilon) {
    std::cout << "ERROR: L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
  } else {
    std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
    std::cout << "L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
#endif
    const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
    size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
    auto avgtime = stencil_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenCL stencil execution on 2D grid" << std::endl;

  prk::opencl::listPlatforms();

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, n, radius, tile_size;
  bool star = true;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <array dimension> [<tile_size> <star/grid> <radius>]";
      }

      // number of times to run the algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // linear grid dimension
      n  = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimension must be positive";
      } else if (n > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

      // default tile size for tiling of local transpose
      tile_size = 32;
      if (argc > 3) {
          tile_size = std::atoi(argv[3]);
          if (tile_size <= 0) tile_size = n;
          if (tile_size > n) tile_size = n;
      }

      // stencil pattern
      if (argc > 4) {
          auto stencil = std::string(argv[4]);
          auto grid = std::string("grid");
          star = (stencil == grid) ? false : true;
      }

      // stencil radius
      radius = 2;
      if (argc > 5) {
          radius = std::atoi(argv[5]);
      }

      if ( (radius < 1) || (2*radius+1 > n) ) {
        throw "ERROR: Stencil radius negative or too large";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid size            = " << n << std::endl;
  std::cout << "Tile size            = " << tile_size << std::endl;
  std::cout << "Type of stencil      = " << (star ? "star" : "grid") << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;

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
        run<double>(cpu, iterations, n, radius, star);
    } else {
        run<float>(cpu, iterations, n, radius, star);
    }
  }

  cl::Context gpu(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(gpu) )
  {
    const int precision = prk::opencl::precision(gpu);

    std::cout << "GPU Precision         = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(gpu, iterations, n, radius, star);
    } else {
        run<float>(gpu, iterations, n, radius, star);
    }
  }

  cl::Context acc(CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(acc) )
  {

    const int precision = prk::opencl::precision(acc);

    std::cout << "ACC Precision         = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(acc, iterations, n, radius, star);
    } else {
        run<float>(acc, iterations, n, radius, star);
    }
  }

  return 0;
}
