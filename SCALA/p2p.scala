// based on Java version
object p2p
{
    def main(args: Array[String]) : Unit = {
        println("Parallel Research Kernels.")
        println("Scala pipeline execution on 2D grid.")

        /*******************************************************************
        **read and test input parameters
        *******************************************************************/

        if (args.length != 3) {
            println("argument count = " + args.length)
            println("Usage: java p2p <# iterations> <first array dimension> <second array dimension>")
            System.exit(0)
        }

        val iterations = args(0).toInt
        if (iterations < 1) {
            println("ERROR: iterations must be >= 1")
            System.exit(0)
        }

        val m = args(1).toInt
        if (m < 1) {
            println("ERROR: array dimension must be >= 1")
            System.exit(0)
        }

        val n = args(2).toInt
        if (n < 1) {
            println("ERROR: array dimension must be >= 1")
            System.exit(0)
        }

        println(f"Grid sizes               = ${m} * ${n}")
        println(f"Number of iterations     = ${iterations}")

        val grid = Array.ofDim[Double](m, n)
        for (j <- 0 until n) grid(0)(j) = j
        for (i <- 0 until m) grid(i)(0) = i

        var startTime = System.nanoTime()
        for (k <- 0 until iterations+1) {
            // start timer after a warmup iteration
            if (k<1)
                startTime = System.nanoTime()
            for (i <- 1 until m) {
                for (j <- 1 until n) {
                    grid(i)(j) = grid(i-1)(j) + grid(i)(j-1) - grid(i-1)(j-1)
                }
            }
            // copy top right corner value to bottom left corner to create dependency
            grid(0)(0) = -grid(m-1)(n-1)
        }

        val pipelineTime = (System.nanoTime() - startTime)*1e-9.toDouble

        /*******************************************************************
        **Final analysis
        *******************************************************************/

        val cornerValue = (iterations+1).toDouble*(m+n-2).toDouble
        val epsilon = 1e-8.toDouble
        if ( (math.abs(grid(m-1)(n-1) - cornerValue)/cornerValue) < epsilon ) {
            println("Solution validates")
            val avgtime = pipelineTime/iterations.toDouble
            val rate = (1e-6.toDouble)*2.toDouble*(m-1)*(n-1)/avgtime.toDouble
            println(f"Rate (MFlops/s): ${rate};  Avg time (s): ${avgtime}")
        } else {
            println("ERROR: checksum ${grid(m-1)(n-1)} does not match verification value ${cornerValue}")
        }
    }
}
