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
/// NAME:    dgemm
///
/// PURPOSE: This program measures the time for the dgemm of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          dgemm <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          dgemm worked and timing statistics.
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///          Converted to Go by Jeff Hammond, January 2021.
///
//////////////////////////////////////////////////////////////////////

package main

import (
    "fmt"
    "flag"
    "os"
    "time"
    "math"
    "gonum.org/v1/gonum/mat"
)

func AddOne(i, j int, v float64) float64 {
 return v+1.0
}

func main() {

  fmt.Println("Parallel Research Kernels")
  fmt.Println("Go Dense matrix-matrix multiplication: C = A x B")

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if len(os.Args) < 2 {
      fmt.Println("Usage: go run dgemm.go -i <# iterations> -n <matrix order>")
      os.Exit(1)
  }

  piterations := flag.Int("i", 0, "iterations")
  porder      := flag.Int("n", 0, "matrix order")
  flag.Parse()

  iterations := *piterations
  order      := *porder

  if (iterations < 1) {
      fmt.Println("ERROR: iterations must be >= 1: ", iterations, *piterations)
      os.Exit(1)
  }

  if (order <= 0) {
      fmt.Println("ERROR: vector order must be positive: ", order, *porder)
      os.Exit(1)
  }

  fmt.Println("Number of iterations = ", iterations)
  fmt.Println("Matrix order         = ", order)

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  A := mat.NewDense(order, order, nil)
  B := mat.NewDense(order, order, nil)
  C := mat.NewDense(order, order, nil)
  T := mat.NewDense(order, order, nil)

  for j := int(0); j<order; j++ {
    for i := int(0); i<order; i++ {
      A.Set(j,i,float64(j))
      B.Set(j,i,float64(j))
      C.Set(j,i,0.0)
    }
  }

  var start = time.Now()

  for iter := 0; iter<=iterations; iter++ {

      if iter==1 {
          start = time.Now()
      }

      T.Mul(A,B)
      C.Add(C,T)
  }
  stop := time.Now()

  dgemm_time := stop.Sub(start)

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  forder := float64(order)
  forder3 := forder * forder * forder
  forder2 := (forder-1) * (forder-1)
  reference := 0.25 * forder3 * forder2 * float64(iterations+1)
  var checksum = float64(0)
  for j := int(0); j<order; j++ {
    for i := int(0); i<order; i++ {
      checksum += C.At(i,j)
    }
  }

  epsilon := float64(1.e-8)
  residuum := math.Abs(checksum-reference)/reference
  if residuum < epsilon {
      fmt.Println("Solution validates")
      avgtime := int64(dgemm_time/time.Microsecond) / int64(iterations)
      nbytes  := 2.0 * forder * forder * forder
      fmt.Printf("Rate (MF/s): %f", float64(nbytes) / float64(avgtime) )
      fmt.Printf(" Avg time (s): %f\n", 1.0e-6 * float64(avgtime) )
  } else {
      fmt.Printf("Failed Validation on output array\n")
      fmt.Printf("ERROR: solution did not validate\n")
      for i := int(0); i<order; i++ {
        for j := int(0); j<order; j++ {
          fmt.Printf("%d %d %f %f %f\n", i, j, A.At(i,j), B.At(i,j), C.At(i,j))
        }
      }
      os.Exit(1)
  }
}


