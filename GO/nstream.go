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
///          vectors, and the offset between vectors.
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
///          Converted to Go by Jeff Hammond, July 2020.
///
//////////////////////////////////////////////////////////////////////

package main

import (
    "fmt"
    "flag"
    "os"
    "time"
    "math"
    "unsafe"
)

func main() {

  fmt.Println("Parallel Research Kernels")
  fmt.Println("Go STREAM triad: A = B + scalar * C")

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if len(os.Args) < 2 {
      fmt.Println("Usage: go run nstream.go -i <# iterations> -n <vector length>")
      os.Exit(1)
  }

  piterations := flag.Int("i", 0, "iterations")
  plength     := flag.Int64("n", 0, "length of vector")
  flag.Parse()

  iterations := *piterations
  length     := *plength

  if (iterations < 1) {
      fmt.Println("ERROR: iterations must be >= 1: ", iterations, *piterations)
      os.Exit(1)
  }

  if (length <= 0) {
      fmt.Println("ERROR: vector length must be positive: ", length, *plength)
      os.Exit(1)
  }

  fmt.Println("Number of iterations = ", iterations)
  fmt.Println("Vector length        = ", length)

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  A := make([]float64, length)
  B := make([]float64, length)
  C := make([]float64, length)

  for i := int64(0); i<length; i++ {
      A[i] = 0
      B[i] = 2
      C[i] = 2
  }

  scalar := float64(3)

  var start = time.Now()

  for iter := 0; iter<=iterations; iter++ {

      if iter==1 {
          start = time.Now()
      }

      for i := int64(0); i<length; i++ {
          A[i] += B[i] + scalar * C[i]
      }
  }
  stop := time.Now()

  nstream_time := stop.Sub(start)

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  ar := float64(0)
  br := float64(2)
  cr := float64(2)
  for i := 0; i<=iterations; i++ {
      ar += br + scalar * cr
  }

  ar *= float64(length)

  asum := float64(0)
  for i := int64(0); i<length; i++ {
      asum += math.Abs(A[i])
  }

  epsilon := float64(1.e-8)
  if math.Abs(ar-asum)/asum > epsilon {
      fmt.Printf("Failed Validation on output array\n")
      fmt.Printf("       Expected checksum: %f\n", ar)
      fmt.Printf("       Observed checksum: %f\n", asum)
      fmt.Printf("ERROR: solution did not validate\n")
      os.Exit(1)
  } else {
      fmt.Println("Solution validates")
      avgtime := int64(nstream_time/time.Microsecond) / int64(iterations)
      nbytes  := int64(4) * length * int64(unsafe.Sizeof(A[0]))
      fmt.Printf("Rate (MB/s): %f", float64(nbytes) / float64(avgtime) )
      fmt.Printf(" Avg time (s): %f\n", 1.0e-6 * float64(avgtime) )
  }
}


