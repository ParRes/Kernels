//
// icpx -std=c++17 -pthread -g -O3 -xHOST -fiopenmp -fopenmp-targets=spir64 nstream-openmp-target-bug.cc
//
#include <cstdio>
#include <cstdlib> // atoi, getenv
#include <cstdint>

#ifdef BUG
#include <cmath>   // abs, fabs
#endif

int main(int argc, char * argv[])
{
  size_t length = 100000;

  double * A = new double[length];
  double * B = new double[length];
  double * C = new double[length];

#pragma omp target data map(tofrom: A[0:length]) map(to: B[0:length], C[0:length])
  {
#pragma omp target teams distribute parallel for
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + C[i];
      }
  }

  return 0;
}


