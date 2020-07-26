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

#include "prk_sycl.h"
#include "prk_util.h"

int main(int argc, char * argv[])
{
  auto qs = prk::SYCL::queues();

  size_t length, local_length;
  int use_ngpu = 1;
  try {
      if (argc < 2) {
        throw "Usage: <vector length> [<use_ngpu>]";
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
  std::cout << "Vector length         = " << length << std::endl;
  std::cout << "Vector length (local) = " << local_length << std::endl;

  int np = use_ngpu;

  auto h_A = prk::vector<double>(length, 0);
  auto h_B = prk::vector<double>(length, 2);

  auto d_A = std::vector<double*> (np, nullptr);
  auto d_B = std::vector<double*> (np, nullptr);

  qs.allocate<double>(d_A, local_length);
  qs.allocate<double>(d_B, local_length);
  qs.waitall();

  qs.scatter<double>(d_A, h_A, local_length);
  qs.scatter<double>(d_B, h_B, local_length);
  qs.waitall();

  // overwrite host buffer with garbage to detect bugs
  h_A.fill(-77777777);

  const double scalar(3);
  {
      for (int iter = 0; iter<=iterations; iter++) {

        if (iter==1) nstream_time = prk::wtime();

        for (int g=0; g<np; ++g) {
            auto q = qs.queue(g);

            auto p_A = d_A[g];
            auto p_B = d_B[g];

            const size_t size  = local_length;

            q.submit([&](sycl::handler& h) {
              h.parallel_for( sycl::range<1>{size}, [=] (sycl::id<1> i) {
                  p_A[i] += p_B[i] + scalar * p_C[i];
              });
            });
        }
        qs.waitall();
      }
      nstream_time = prk::wtime() - nstream_time;
  }

  qs.gather<double>(h_A, d_A, local_length);
  qs.gather<double>(h_B, d_B, local_length);
  qs.waitall();

  qs.free(d_A);
  qs.free(d_B);
  qs.waitall();

  return 0;
}


