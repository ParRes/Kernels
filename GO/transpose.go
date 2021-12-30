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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
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
    "unsafe"
    "gonum.org/v1/gonum/mat"
)

func tselect(t, o int) int {
    if (t == 0) {
        return o
    } else {
        return t
    }
}

func AddOne(i, j int, v float64) float64 {
 return v+1.0
}

func main() {

  fmt.Println("Parallel Research Kernels")
  fmt.Println("Go Matrix transpose: B = A^T")

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if len(os.Args) < 2 {
      fmt.Println("Usage: go run transpose.go -i <# iterations> -n <matrix order> [-t <tile size>]")
      os.Exit(1)
  }

  piterations := flag.Int("i", 0, "iterations")
  porder      := flag.Int("n", 0, "matrix order")
  ptilesize   := flag.Int("t", 0, "tile size")
  flag.Parse()

  iterations := *piterations
  order      := *porder
  tilesize   := tselect(*ptilesize,*porder)

  if (iterations < 1) {
      fmt.Println("ERROR: iterations must be >= 1: ", iterations, *piterations)
      os.Exit(1)
  }

  if (order <= 0) {
      fmt.Println("ERROR: vector order must be positive: ", order, *porder)
      os.Exit(1)
  }

  if (tilesize > order || tilesize < 1) {
      fmt.Println("ERROR: tilesize must be between 1 and order: ", tilesize, *ptilesize)
      os.Exit(1)
  }

  fmt.Println("Number of iterations = ", iterations)
  fmt.Println("Matrix order         = ", order)
  if (tilesize > 0) {
      fmt.Println("Tile size            = ", tilesize)
  } else {
      fmt.Println("Untiled")
  }

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  A := mat.NewDense(order, order, nil)
  B := mat.NewDense(order, order, nil)

  for j := int(0); j<order; j++ {
    for i := int(0); i<order; i++ {
      A.Set(j,i,float64(i*order+j))
      B.Set(j,i,0.0)
    }
  }

  scalar := float64(3)

  var start = time.Now()

  for iter := 0; iter<=iterations; iter++ {

      if iter==1 {
          start = time.Now()
      }

      // The T() method is inadequate...
      //B += A.T()
      // Implements A += 1.0
      //A.Apply(AddOne,A)

      // This is such an embarrassing implementation...
      for i := int(0); i<order; i++ {
        for j := int(0); j<order; j++ {
          B.Set(i,j,B.At(i,j)+A.At(j,i))
          A.Set(j,i,A.At(j,i)+1.0)
        }
      }
  }
  stop := time.Now()

  transpose_time := stop.Sub(start)

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  addit := float64(((iterations+1)*iterations)/2)
  var abserr = float64(0)
  for j := int(0); j<order; j++ {
    for i := int(0); i<order; i++ {
      ji := j*order+i
      reference := float64(ji)*float64(1+iterations)+addit
      abserr += math.Abs(B.At(j,i) - reference)
    }
  }

  epsilon := float64(1.e-8)
  if abserr < epsilon {
      fmt.Println("Solution validates")
      avgtime := int64(transpose_time/time.Microsecond) / int64(iterations)
      nbytes  := 2 * order * order * int(unsafe.Sizeof(scalar))
      fmt.Printf("Rate (MB/s): %f", float64(nbytes) / float64(avgtime) )
      fmt.Printf(" Avg time (s): %f\n", 1.0e-6 * float64(avgtime) )
  } else {
      fmt.Printf("Failed Validation on output array\n")
      fmt.Printf("ERROR: solution did not validate\n")
      for i := int(0); i<order; i++ {
        for j := int(0); j<order; j++ {
          fmt.Printf("%d %d %f %f\n", i, j, A.At(i,j), B.At(i,j))
        }
      }
      os.Exit(1)
  }
}


