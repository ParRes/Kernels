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
///          transpose <matrix_size> <# iterations>
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "CL/sycl.hpp"
#include "prk_util.h"

#if 0
#include "prk_opencl.h"
#define USE_OPENCL 1
#endif

template <typename T> class transpose;

template <typename T>
void run(cl::sycl::queue & q, int iterations, size_t order)
{
  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  double trans_time(0);

  std::vector<T> h_A(order*order);
  std::vector<T> h_B(order*order,(T)0);

  // fill A with the sequence 0 to order^2-1 as doubles
  std::iota(h_A.begin(), h_A.end(), static_cast<T>(0));

  try {

#if PREBUILD_KERNEL
    cl::sycl::program kernel(q.get_context());
    kernel.build_with_kernel_type<transpose<T>>();
#endif

#if USE_2D_INDEXING
    cl::sycl::buffer<T,2> d_A( h_A.data(), cl::sycl::range<2>{order,order} );
    cl::sycl::buffer<T,2> d_B( h_B.data(), cl::sycl::range<2>{order,order} );
#else
    cl::sycl::buffer<T> d_A { h_A.data(), h_A.size() };
    cl::sycl::buffer<T> d_B { h_B.data(), h_B.size() };
#endif

    for (int iter = 0; iter<=iterations; ++iter) {

      if (iter==1) trans_time = prk::wtime();

      q.submit([&](cl::sycl::handler& h) {

        // accessor methods
        auto A = d_A.template get_access<cl::sycl::access::mode::read_write>(h);
        auto B = d_B.template get_access<cl::sycl::access::mode::read_write>(h);

        h.parallel_for<class transpose<T>>(
#if PREBUILD_KERNEL
                kernel.get_kernel<transpose<T>>(),
#endif
                cl::sycl::range<2>{order,order}, [=] (cl::sycl::item<2> it) {
#if USE_2D_INDEXING
          cl::sycl::id<2> ij{it[0],it[1]};
          cl::sycl::id<2> ji{it[1],it[0]};
          B[ij] += A[ji];
          A[ji] += (T)1;
#else
          B[it[0] * order + it[1]] += A[it[1] * order + it[0]];
          A[it[1] * order + it[0]] += (T)1;
#endif
        });
      });
      q.wait();
    }

    // Stop timer before buffer+accessor destructors fire,
    // since that will move data, and we do not time that
    // for other device-oriented programming models.
    trans_time = prk::wtime() - trans_time;
  }
  catch (cl::sycl::exception & e) {
    std::cout << e.what() << std::endl;
#ifdef __COMPUTECPP__
    std::cout << e.get_file_name() << std::endl;
    std::cout << e.get_line_number() << std::endl;
    std::cout << e.get_description() << std::endl;
    std::cout << e.get_cl_error_message() << std::endl;
    std::cout << e.get_cl_code() << std::endl;
#endif
    return;
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
    return;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  // TODO: replace with std::generate, std::accumulate, or similar
  const T addit = (iterations+1.) * (iterations/2.);
  double abserr(0);
  for (size_t i=0; i<order; ++i) {
    for (size_t j=0; j<order; ++j) {
      size_t const ij = i*order+j;
      size_t const ji = j*order+i;
      const T reference = static_cast<T>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(h_B[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const double epsilon(1.0e-8);
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    double avgtime = trans_time/iterations;
    double bytes = (size_t)order * (size_t)order * sizeof(T);
    std::cout << 8*sizeof(T) << "B "
              << "Rate (MB/s): " << 1.0e-6 * (2.*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
  }
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/SYCL Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t order;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order>";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      } else if (order > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup SYCL environment
  //////////////////////////////////////////////////////////////////////

#ifdef USE_OPENCL
  prk::opencl::listPlatforms();
#endif

  try {
#if SYCL_TRY_CPU_QUEUE
    if (1) {
        cl::sycl::queue host(cl::sycl::host_selector{});
#ifndef TRISYCL
        auto device      = host.get_device();
        auto platform    = device.get_platform();
        std::cout << "SYCL Device:   " << device.get_info<cl::sycl::info::device::name>() << std::endl;
        std::cout << "SYCL Platform: " << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
#endif
        run<float>(host, iterations, order);
        run<double>(host, iterations, order);
    }
#endif

    // CPU requires spir64 target
#if SYCL_TRY_CPU_QUEUE
    if (1) {
        cl::sycl::queue cpu(cl::sycl::cpu_selector{});
#if !defined(TRISYCL) && !defined(__HIPSYCL__)
        auto device      = cpu.get_device();
        auto platform    = device.get_platform();
        std::cout << "SYCL Device:   " << device.get_info<cl::sycl::info::device::name>() << std::endl;
        std::cout << "SYCL Platform: " << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
        bool has_spir = device.has_extension(cl::sycl::string_class("cl_khr_spir"));
#else
        bool has_spir = true; // ?
#endif
        if (has_spir) {
          run<float>(cpu, iterations, order);
          run<double>(cpu, iterations, order);
        }
    }
#endif

    // NVIDIA GPU requires ptx64 target and does not work very well
#if SYCL_TRY_GPU_QUEUE
    if (0) {
        cl::sycl::queue gpu(cl::sycl::gpu_selector{});
#if !defined(TRISYCL) && !defined(__HIPSYCL__)
        auto device      = gpu.get_device();
        auto platform    = device.get_platform();
        std::cout << "SYCL Device:   " << device.get_info<cl::sycl::info::device::name>() << std::endl;
        std::cout << "SYCL Platform: " << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
        bool has_spir = device.has_extension(cl::sycl::string_class("cl_khr_spir"));
        bool has_fp64 = device.has_extension(cl::sycl::string_class("cl_khr_fp64"));
#else
        bool has_spir = true; // ?
        bool has_fp64 = true;
#endif
        if (!has_fp64) {
          std::cout << "SYCL GPU device lacks FP64 support." << std::endl;
        }
        if (has_spir) {
          run<float>(gpu, iterations, order);
          if (has_fp64) {
            run<double>(gpu, iterations, order);
          }
        } else {
          std::cout << "SYCL GPU device lacks SPIR-V support." << std::endl;
#ifdef __COMPUTECPP__
          std::cout << "You are using ComputeCpp so we will try it anyways..." << std::endl;
          run<float>(gpu, iterations, order);
          if (has_fp64) {
            run<double>(gpu, iterations, order);
          }
#endif
        }
    }
#endif
  }
  catch (cl::sycl::exception & e) {
    std::cout << e.what() << std::endl;
#ifdef __COMPUTECPP__
    std::cout << e.get_file_name() << std::endl;
    std::cout << e.get_line_number() << std::endl;
    std::cout << e.get_description() << std::endl;
    std::cout << e.get_cl_error_message() << std::endl;
    std::cout << e.get_cl_code() << std::endl;
#endif
    return 1;
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  return 0;
}


