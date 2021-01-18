object nstream
{
    def main (args: Array[String])
    {
        println("Parallel Research Kernels")
        println("Scala Stream triad: A = B + scalar * C.")

        /////////////////////////////////////////////////////////
        // read and test input parameters
        /////////////////////////////////////////////////////////

        if (args.length != 2){
            println("Usage: java nstream <# iterations> <vector length>")
            return
        }

        val iterations = args(0).toInt
        if (iterations < 1) {
            println("ERROR: iterations must be >= 1")
            return
        }

        val length = args(1).toInt
        if (length < 1) {
            println("ERROR: vector length must be positive")
            return
        }

        println("Vector length        = " + length)
        println("Number of iterations = " + iterations)

        var a = Array.ofDim[Double](length)
        var b = Array.ofDim[Double](length)
        var c = Array.ofDim[Double](length)

        for (j <- 1 to length) {
            a(j) = 0.0
            b(j) = 2.0
            c(j) = 2.0
        }

        val scalar: Double = 3

        var t0 = System.nanoTime()

        for (iter <- 0 to iterations) {

            /* start timer after a warmup iteration */
            if (iter == 1) {
                t0 = System.nanoTime()
            }

            for (j <- 1 to length) {
                a(j) += b(j) + scalar*c(j)
            }
        }

        val t1 = System.nanoTime()
        val nstream_time = t1 - t0

        /////////////////////////////////////////////////////////
        // Analyze and output results.
        /////////////////////////////////////////////////////////

        var ar : Double = 0
        val br : Double = 2
        val cr : Double = 2

        for (k <- 0 to iterations) {
            ar = ar + br + scalar * cr
        }

        ar *= length;

        var asum : Double = 0
        for (i <- 1 to length) {
            asum += Math.abs(a(i))
        }

        val epsilon : Double = 1.0E-8
        if (Math.abs(ar-asum)/asum > epsilon) {
            println("Failed Validation on output array")
            println("        Expected checksum: " + ar)
            println("        Observed checksum: " + asum)
            println("ERROR: solution did not validate")
        } else {
            println("Solution validates")
            val avgtime = nstream_time/iterations/1000
            val nbytes  = 4 * length * 8
            println("Rate (MB/s): " + 1.0E-6*nbytes/avgtime + " Avg time (s): " + avgtime)
        }
    }
}
