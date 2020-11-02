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

#include "prk_sycl.h"
#include "prk_util.h"

template <typename T> class transpose;

template <typename T>
void run(sycl::queue & q, int iterations, size_t order, size_t block_size)
{
  size_t padded_order = block_size * prk::divceil(order,block_size);
  sycl::range<2> global{padded_order,padded_order};
  sycl::range<2> local{block_size,block_size};

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  double trans_time{0};

  std::vector<T> h_A(order*order);
  std::vector<T> h_B(order*order,(T)0);

  // fill A with the sequence 0 to order^2-1 as doubles
  std::iota(h_A.begin(), h_A.end(), static_cast<T>(0));

  try {

#if PREBUILD_KERNEL
    auto ctx = q.get_context();
    sycl::program kernel{ctx};
    kernel.build_with_kernel_type<transpose<T>>();
#endif

    sycl::buffer<T,2> d_A( h_A.data(), sycl::range<2>{order,order} );
    sycl::buffer<T,2> d_B( h_B.data(), sycl::range<2>{order,order} );

    for (int iter = 0; iter<=iterations; ++iter) {

      if (iter==1) trans_time = prk::wtime();

      q.submit([&](sycl::handler& h) {

        // accessor methods
        auto A = d_A.template get_access<sycl::access::mode::read_write>(h);
        auto B = d_B.template get_access<sycl::access::mode::read_write>(h);

        h.parallel_for<class transpose<T>>(
#if PREBUILD_KERNEL
            kernel.get_kernel<transpose<T>>(),
#endif
            sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
                const size_t i = it.get_global_id(0);
                const size_t j = it.get_global_id(1);
                if ((i<order) && (j<order)) {
                    sycl::id<2> ij{i,j};
                    sycl::id<2> ji{j,i};
                    B[ij] += A[ji];
                    A[ji] += (T)1;
                }
        });
      });
      q.wait();
    }
    // Stop timer before buffer+accessor destructors fire,
    // since that will move data, and we do not time that
    // for other device-oriented programming models.
    trans_time = prk::wtime() - trans_time;
  }
  catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
    prk::SYCL::print_exception_details(e);
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

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr(0);
  for (size_t i=0; i<order; ++i) {
    for (size_t j=0; j<order; ++j) {
      const size_t ij = i*order+j;
      const size_t ji = j*order+i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += prk::abs(h_B[ji] - reference);
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
  size_t order, block_size;

  block_size = 16;

  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> [<block_size>]";
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
      } else if (order > prk::get_max_matrix_size()) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }

      if (argc>3) {
         block_size = std::atoi(argv[3]);
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Block size            = " << block_size << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup SYCL environment
  //////////////////////////////////////////////////////////////////////

  try {
    sycl::queue q{sycl::host_selector{}};
    prk::SYCL::print_device_platform(q);
    run<float>(q, iterations, order, block_size);
    run<double>(q, iterations, order, block_size);
  }
  catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
    prk::SYCL::print_exception_details(e);
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
  }

  try {
    sycl::queue q{sycl::cpu_selector{}};
    prk::SYCL::print_device_platform(q);
    run<float>(q, iterations, order, block_size);
    run<double>(q, iterations, order, block_size);
  }
  catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
    prk::SYCL::print_exception_details(e);
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
  }

  try {
    sycl::queue q{sycl::gpu_selector{}};
    prk::SYCL::print_device_platform(q);
    bool has_fp64 = prk::SYCL::has_fp64(q);
    run<float>(q, iterations, order, block_size);
    if (has_fp64) {
      run<double>(q, iterations, order, block_size);
    } else {
      std::cout << "SYCL GPU device lacks FP64 support." << std::endl;
    }
  }
  catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
    prk::SYCL::print_exception_details(e);
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
  }

  return 0;
}


