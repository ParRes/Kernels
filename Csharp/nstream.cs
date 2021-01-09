using System;

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



    }

  } // nstream

} // PRK
