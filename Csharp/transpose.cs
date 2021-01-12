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
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///          Converted to C# by Jeff Hammond, January 2021.
///
//////////////////////////////////////////////////////////////////////

using System;
using System.Diagnostics;

namespace PRK {

  class transpose {

    static void Help() {
      Console.WriteLine("Usage: <# iterations> <vector length>");
    }

    static void Main(string[] args)
    {
      Console.WriteLine("Parallel Research Kernels version ");
      Console.WriteLine("C# Matrix transpose: B = A^T");

      //////////////////////////////////////////////////////////////////////
      // Read and test input parameters
      //////////////////////////////////////////////////////////////////////

      if (args.Length != 2) {
          Help();
          System.Environment.Exit(args.Length+1);
      }

      // This requires a newer C# compiler than 4.6 (e.g. 6.8)
      //if ( int.TryParse(args[0], out int iterations) ) {
      //    Console.WriteLine("Number of iterations = {0}", iterations);
      //} else {
      //    Help();
      //}

      int iterations = int.Parse(args[0]);
      int order      = int.Parse(args[1]);

      //////////////////////////////////////////////////////////////////////
      // Allocate space for the input and transpose matrix
      //////////////////////////////////////////////////////////////////////

      double[] A = new double[order*order];
      double[] B = new double[order*order];

      Stopwatch timer = new Stopwatch();

      for (int k = 0 ; k <= iterations ; k++) {

          if (k == 0) {
              timer.Start();
          }

          for (int i = 0 ; i < order ; i++) {
              for (int j = 0 ; j < order ; j++) {
                  B[i*order+j] += A[j*order+i];
                  A[j*order+i] += 1.0;
              }
          }
      }
      timer.Stop();
      long tics = timer.ElapsedTicks;
      long freq = Stopwatch.Frequency;
      double trans_time = (double)tics/(double)freq;

      //////////////////////////////////////////////////////////////////////
      // Analyze and output results
      //////////////////////////////////////////////////////////////////////

      double addit = (iterations+1) * iterations / 2;
      double abserr = 0.0;
      for (int j=0; j<order; j++) {
        for (int i=0; i<order; i++) {
          long ij = i*order+j;
          long ji = j*order+i;
          double reference = (double)(ij)*(1+iterations)+addit;
          abserr += Math.Abs(B[ji] - reference);
        }
      }

      const double epsilon = 1.0e-8;
      if (abserr < epsilon) {
        Console.WriteLine("Solution validates");
        double avgtime = trans_time/iterations;
        double bytes = order * order * sizeof(double);
        Console.WriteLine("Rate (MB/s): {0} Avg time (s): {1}", 2.0e-6 * bytes/avgtime, avgtime );
      } else {
        Console.WriteLine("ERROR: Aggregate squared error {0} exceeds threshold {1}", abserr, epsilon );
      }

    }

  } // transpose

} // PRK
