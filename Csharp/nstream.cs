using System;
using System.Diagnostics;

namespace PRK {

  class nstream {

    static void Help() {
      Console.WriteLine("Usage: <# iterations> <vector length>");
    }

    static void Main(string[] args) {
      Console.WriteLine("Parallel Research Kernels");
      Console.WriteLine("Rust STREAM triad: A = B + scalar * C");

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
      double dt = (double)tics/(double)freq;
      Console.WriteLine("tics={0} freq={1} dt={2}", tics, freq, dt);

    }

  } // nstream

} // PRK
