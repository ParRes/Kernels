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
      Console.WriteLine("Rust STREAM triad: A = B + scalar * C");

      //////////////////////////////////////////////////////////////////////
      // Read and test input parameters
      //////////////////////////////////////////////////////////////////////

      if (args.Length != 2) {
          Help();
          System.Environment.Exit(args.Length+1);
      }

      if ( int.TryParse(args[0], out int iterations) ) {
          Console.WriteLine("Number of iterations  = {0}", iterations);
      } else {
          Help();
      }

      if ( int.TryParse(args[1], out int length) ) {
          Console.WriteLine("vector length         = {0}", length);
      } else {
          Help();
      }


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

      //System.Diagnostics.Stopwatch nstream_timer;
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
      //Console.WriteLine("tics={0} freq={1} dt={2}", tics, freq, nstream_time);

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
