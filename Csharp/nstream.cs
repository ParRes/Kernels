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
///          of iterations to loop over the triad vectors and
///          the length of the vectors.
///
///          <progname> <# iterations> <vector length>
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
///          Converted to C# by Jeff Hammond, January 2021.
///
//////////////////////////////////////////////////////////////////////

using System;
using System.Diagnostics;

namespace PRK {

  class nstream {

    static void Help() {
      Console.WriteLine("Usage: <# iterations> <vector length>");
    }

    static void Main(string[] args)
    {
      Console.WriteLine("Parallel Research Kernels");
      Console.WriteLine("C# STREAM triad: A = B + scalar * C");

      //////////////////////////////////////////////////////////////////////
      // Read and test input parameters
      //////////////////////////////////////////////////////////////////////

      if (args.Length != 2) {
          Help();
          System.Environment.Exit(args.Length+1);
      }

      int iterations = int.Parse(args[0]);
      int length     = int.Parse(args[1]);

      //////////////////////////////////////////////////////////////////////
      // Allocate space and perform the computation
      //////////////////////////////////////////////////////////////////////

      double[] A = new double[length];
      double[] B = new double[length];
      double[] C = new double[length];

      for (int i = 0 ; i < length ; i++) {
          A[i] = 0.0;
          B[i] = 2.0;
          C[i] = 2.0;
      }

      double scalar = 3.0;

      Stopwatch timer = new Stopwatch();

      for (int k = 0 ; k <= iterations ; k++) {

          if (k == 0) {
              timer.Start();
          }

          for (int i = 0 ; i < length ; i++) {
              A[i] += B[i] + scalar * C[i];
          }
      }
      timer.Stop();
      long tics = timer.ElapsedTicks;
      long freq = Stopwatch.Frequency;
      double nstream_time = (double)tics/(double)freq;

      //////////////////////////////////////////////////////////////////////
      // Analyze and output results
      //////////////////////////////////////////////////////////////////////

      double ar = 0.0;
      double br = 2.0;
      double cr = 2.0;
      for (int k = 0 ; k <= iterations ; k++) {
          ar += br + scalar * cr;
      }

      ar *= length;

      double asum = 0.0;
      for (int i = 0 ; i < length ; i++) {
          asum += Math.Abs(A[i]);
      }

      const double epsilon=1e-8;
      if (Math.Abs(ar-asum)/asum > epsilon) {
          Console.WriteLine("Failed Validation on output array");
          Console.WriteLine("       Expected checksum: {0}",ar);
          Console.WriteLine("       Observed checksum: {0}",asum);
          Console.WriteLine("ERROR: solution did not validate");
      } else {
          Console.WriteLine("Solution validates");
          double avgtime = nstream_time/iterations;
          double nbytes = 4.0 * length * sizeof(double);
          Console.WriteLine("Rate (MB/s): {0} Avg time (s): {1}", 1e-6*nbytes/avgtime, avgtime);
      }

    }

  } // nstream

} // PRK
