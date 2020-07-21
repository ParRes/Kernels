///
/// Copyright (c) 2020, Intel Corporation
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

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/DPC++ STREAM triad: A = B + scalar * C" << std::endl;

  std::vector<sycl::queue> qs;

  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") != std::string::npos) {
        std::cout << "*Level Zero GPU skipped" << std::endl;
        break;
    }
    if ( pname.find("Intel") == std::string::npos) {
        std::cout << "*non-Intel skipped" << std::endl;
        break;
    }
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
        if ( d.is_gpu() || d.is_cpu() ) {
            std::cout << "**Device is CPU or GPU - adding to vector of queues" << std::endl;
            qs.push_back(sycl::queue(d));
        }
    }
  }

  int haz_ngpu = qs.size();
  std::cout << "Number of CPUs and GPUs found  = " << haz_ngpu << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t length, local_length;
  int use_ngpu = 1;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length> [<use_ngpu>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atoi(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }

      if (argc > 3) {
        use_ngpu = std::atoi(argv[3]);
      }
      if ( use_ngpu > qs.size() ) {
          std::string error = "You cannot use more devices ("
                            + std::to_string(use_ngpu)
                            + ") than you have ("
                            + std::to_string(qs.size()) + ")";
          throw error;
      }

      if (length % use_ngpu != 0) {
          std::string error = "ERROR: vector length ("
                            + std::to_string(length)
                            + ") should be divisible by # procs ("
                            + std::to_string(use_ngpu) + ")";
          throw error;
      }
      local_length = length / use_ngpu;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of devices     = " << use_ngpu << std::endl;
  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Vector length         = " << length << std::endl;
  std::cout << "Vector length (local) = " << local_length << std::endl;

  int np = use_ngpu;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time(0);

  auto h_A = prk::vector<double>(length, 0);
  auto h_B = prk::vector<double>(length, 2);
  auto h_C = prk::vector<double>(length, 2);

  auto d_A = std::vector<double*> (np, nullptr);
  auto d_B = std::vector<double*> (np, nullptr);
  auto d_C = std::vector<double*> (np, nullptr);

#if 0
  for (int g=0; g<np; ++g) {
      auto q = qs[g];
      const auto local_bytes = local_length * sizeof(double);
      d_A[g] = syclx::malloc_device<double>(local_length, q);
      d_B[g] = syclx::malloc_device<double>(local_length, q);
      d_C[g] = syclx::malloc_device<double>(local_length, q);
      q.wait();
  }
#else
  auto list(qs);
  for (const auto & l : list | boost::adaptors::indexed(0) ) {
      auto i = l.index();
      auto v = l.value();
      d_A[i] = syclx::malloc_device<double>(local_length, v);
      d_B[i] = syclx::malloc_device<double>(local_length, v);
      d_C[i] = syclx::malloc_device<double>(local_length, v);
  }
  for (auto & i : list) {
      i.wait();
  }
#endif

#if 0
  for (int g=0; g<np; ++g) {
      auto q = qs[g];
      const size_t start = local_length * g;
      const size_t size  = local_length * sizeof(double);
      q.memcpy(d_A[g], &(h_A[start]), size);
      q.memcpy(d_B[g], &(h_B[start]), size);
      q.memcpy(d_C[g], &(h_C[start]), size);
  }
#elif 0
  for (const auto & l : list | boost::adaptors::indexed(0) ) {
      auto i = l.index();
      auto v = l.value();
      auto start = local_length * i;
      auto bytes = local_length * sizeof(double);
      v.memcpy(d_A[i], &(h_A[start]), bytes);
      v.memcpy(d_B[i], &(h_B[start]), bytes);
      v.memcpy(d_C[i], &(h_C[start]), bytes);
  }
#else
  for (const auto & l : list | boost::adaptors::indexed(0) ) {
      auto i = l.index();
      auto v = l.value();
      auto bytes = local_length * sizeof(double);
      auto start = local_length * i;
      {
          auto target = d_A[i];
          auto source = &h_A[i * local_length];
          v.memcpy(target, source, bytes);
      }
      {
          auto target = d_B[i];
          auto source = &h_B[i * local_length];
          v.memcpy(target, source, bytes);
      }
      {
          auto target = d_C[i];
          auto source = &h_C[i * local_length];
          v.memcpy(target, source, bytes);
      }
  }
#endif
  for (auto & i : list) {
      i.wait();
  }

  // overwrite host buffer with garbage to detect bugs
  h_A.fill(-77777777);

  const double scalar(3);
  {
      for (int iter = 0; iter<=iterations; iter++) {

        if (iter==1) nstream_time = prk::wtime();

        for (int g=0; g<np; ++g) {
            auto q = qs[g];

            auto p_A = d_A[g];
            auto p_B = d_B[g];
            auto p_C = d_C[g];

            const size_t size  = local_length;

            q.submit([&](sycl::handler& h) {
              h.parallel_for( sycl::range<1>{size}, [=] (sycl::id<1> i) {
                  p_A[i] += p_B[i] + scalar * p_C[i];
              });
            });
        }
        for (auto & q : qs) {
            q.wait();
        }
      }
      nstream_time = prk::wtime() - nstream_time;
  }

  for (const auto & l : list | boost::adaptors::indexed(0) ) {
      auto i = l.index();
      auto v = l.value();
      auto bytes = local_length * sizeof(double);
      auto start = local_length * i;
      auto target = &h_A[i * local_length];
      auto source = d_A[i];
      v.memcpy(target, source, bytes);
  }
  for (auto & q : qs) {
      q.wait();
  }

#if 0
  for (int g=0; g<np; ++g) {
      auto q = qs[g];
      syclx::free(d_C[g], q);
      syclx::free(d_B[g], q);
      syclx::free(d_A[g], q);
  }
#else
  for (const auto & l : list | boost::adaptors::indexed(0) ) {
      auto i = l.index();
      auto v = l.value();
      syclx::free(d_A[i], v);
      syclx::free(d_B[i], v);
      syclx::free(d_C[i], v);
  }
#endif
  for (auto & q : qs) {
      q.wait();
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  double br(2);
  double cr(2);
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (int i=0; i<length; i++) {
    asum += prk::abs(h_A[i]);
  }

  double epsilon=1.e-8;
  if (prk::abs(ar - asum) / asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
      return 1;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      std::cout << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}


