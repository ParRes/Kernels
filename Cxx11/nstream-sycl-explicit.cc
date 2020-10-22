///
/// Copyright (c) 2017, Intel Corporation
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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors, the length of the
///          vectors, and the offset between vectors
///
///          <progname> <# iterations> <vector length> <offset>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// NOTES:   Bandwidth is determined as the number of words read, plus the
///          number of words written, times the size of the words, divided
///          by the execution time. For a vector length of N, the total
///          number of words read and written is 4*N*sizeof(double).
///
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_sycl.h"
#include "prk_util.h"

template <typename T> class nstream;

template <typename T>
void run(sycl::queue & q, int iterations, size_t length, size_t block_size)
{
  sycl::range global{length};
  sycl::range local{block_size};

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time(0);

  const T scalar(3);

  std::vector<T> h_A(length,0);

  try {

#if PREBUILD_KERNEL
    auto ctx = q.get_context();
    sycl::program kernel(ctx);
    kernel.build_with_kernel_type<nstream<T>>();
#endif

    sycl::buffer<T> d_A { sycl::range<1>{length} };
    sycl::buffer<T> d_B { sycl::range<1>{length} };
    sycl::buffer<T> d_C { sycl::range<1>{length} };

    q.submit([&](sycl::handler& h) {
        sycl::accessor<T, 1, sycl::access::mode::write, sycl::access::target::global_buffer> A(d_A, h, sycl::range<1>(length), sycl::id<1>(0));
        h.fill(A,(T)0);
    });
    q.submit([&](sycl::handler& h) {
        sycl::accessor<T, 1, sycl::access::mode::write, sycl::access::target::global_buffer> B(d_B, h, sycl::range<1>(length), sycl::id<1>(0));
        h.fill(B,(T)2);
    });
    q.submit([&](sycl::handler& h) {
        sycl::accessor<T, 1, sycl::access::mode::write, sycl::access::target::global_buffer> C(d_C, h, sycl::range<1>(length), sycl::id<1>(0));
        h.fill(C,(T)2);
    });
    q.wait();

    for (int iter = 0; iter<=iterations; ++iter) {

      if (iter==1) nstream_time = prk::wtime();

      q.submit([&](sycl::handler& h) {

        sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer> A(d_A, h, sycl::range<1>(length), sycl::id<1>(0));
        sycl::accessor<T, 1, sycl::access::mode::read,       sycl::access::target::global_buffer> B(d_B, h, sycl::range<1>(length), sycl::id<1>(0));
        sycl::accessor<T, 1, sycl::access::mode::read,       sycl::access::target::global_buffer> C(d_C, h, sycl::range<1>(length), sycl::id<1>(0));

        h.parallel_for<class nstream<T>>(
#if PREBUILD_KERNEL
                kernel.get_kernel<nstream<T>>(),
#endif
#if 0 // defaults in DPC++ are not optimal for V100
		sycl::range<1>{length}, [=] (sycl::id<1> it) {
		const size_t i = it;
#else
		sycl::nd_range{global, local}, [=](sycl::nd_item<1> it) {
		const size_t i = it.get_global_id(0);
#endif
            A[i] += B[i] + scalar * C[i];
        });
      });
      q.wait();
    }

    // Stop timer before buffer+accessor destructors fire,
    // since that will move data, and we do not time that
    // for other device-oriented programming models.
    nstream_time = prk::wtime() - nstream_time;

    q.submit([&](sycl::handler& h) {
        sycl::accessor<T, 1, sycl::access::mode::read, sycl::access::target::global_buffer> A(d_A, h, sycl::range<1>(length), sycl::id<1>(0));
        h.copy(A,h_A.data());
    });
    q.wait();
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

  double ar(0);
  T br(2);
  T cr(2);
  for (int i=0; i<=iterations; ++i) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (size_t i=0; i<length; ++i) {
      asum += prk::abs(h_A[i]);
  }

  const double epsilon(1.e-8);
  if (prk::abs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(T);
      std::cout << 8*sizeof(T) << "B "
                << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/SYCL STREAM triad: A = B + scalar * C" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, offset;
  size_t length, block_size;

#ifdef DPCPP_CUDA
  block_size = 256; // matches CUDA version default
#else
  block_size = 16384;
#endif

  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length> [<block_size>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atol(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }

      if (argc>3) {
         block_size = std::atoi(argv[3]);
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;
  std::cout << "Block size           = " << block_size << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup SYCL environment
  //////////////////////////////////////////////////////////////////////

  try {
    if (1) {
      sycl::queue q(sycl::host_selector{});
      prk::SYCL::print_device_platform(q);
      run<float>(q, iterations, length, block_size);
      run<double>(q, iterations, length, block_size);
    } else {
        std::cout << "Skipping host device since it is too slow for large problems" << std::endl;
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

  try {
    sycl::queue q(sycl::cpu_selector{});
    prk::SYCL::print_device_platform(q);
    run<float>(q, iterations, length, block_size);
    run<double>(q, iterations, length, block_size);
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
    sycl::queue q(sycl::gpu_selector{});
    prk::SYCL::print_device_platform(q);
    bool has_fp64 = prk::SYCL::has_fp64(q);
    run<float>(q, iterations, length, block_size);
    if (has_fp64) {
      run<double>(q, iterations, length, block_size);
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


