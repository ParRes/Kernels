//
// icx -std=c11 -pthread -g -O3 -xHOST -fiopenmp -fopenmp-targets=spir64 nstream-openmp-target-bug.c
//
#include <stdio.h>
#include <stdlib.h> // atoi, getenv
#include <stdint.h>
#include <math.h>   // abs, fabs

int main(int argc, char * argv[])
{
  size_t length = 100000;

  double * A = malloc(sizeof(double)*length);
  double * B = malloc(sizeof(double)*length);
  double * C = malloc(sizeof(double)*length);

#pragma omp target data map(tofrom: A[0:length]) map(to: B[0:length], C[0:length])
  {
#pragma omp target teams distribute parallel for
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + C[i];
      }
  }

  return 0;
}


