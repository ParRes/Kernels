// based on Java version
object transpose
{
    def main(args: Array[String]) : Unit = {
        println("Parallel Research Kernels.")
        println("Java Matrix transpose: B = A^T.")

        /*******************************************************************
        **read and test input parameters
        *******************************************************************/
        def error_and_exit(s: String) : Unit = {
            println(s)
            System.exit(1)
        }

        if (args.length != 3 && args.length != 2) {
            error_and_exit("Usage: java transpose <# iterations> <matrix order> [tile size]")
        }

        val iterations = args(0).toInt
        if (iterations < 1) {
            error_and_exit("ERROR: iterations must be >= 1")
        }

        val order = args(1).toInt
        if (order < 1) {
            error_and_exit("ERROR: matrix order must be >= 1")
        }

        val tileSize = 
            if (args.length == 3) {
                args(2).toInt
                /* a non-positive tile size means no tiling of the local transpose */
            } else order

        println("Matrix order          = " + order)
        if (tileSize < order)
            println("Tile size             = " + tileSize)
        else
            println("Untiled")
        println("Number of iterations  = " + iterations)


        val A = Array.ofDim[Double](order,order)
        val B = Array.ofDim[Double](order,order)

        for (i <- 0 until order) {
            for (j <- 0 until order) {
                A(i)(j) = i*order+j
            }
        }


        var startTime = System.nanoTime()
        val epsilon = 1.0e-8
        val bytes = 2.0 * 8.0 * order * order

        for (iter <- 0 to iterations) {

            /* start timer after a warmup iteration                                        */
            if (iter == 1)
                startTime = System.nanoTime()

            /* Transpose the  matrix, only use tiling if the tile size is smaller
               than the matrix */
            if (tileSize < order) {
                for (i <- 0 until order by tileSize)
                    for (j <- 0 until order by tileSize)
                        for (it <- i until Math.min(order,i+tileSize))
                            for (jt <- j until Math.min(order,j+tileSize)) {
                                B(it)(jt) += A(jt)(it)
                                A(jt)(it) += 1.0
                            }
            }
            else {
                for (i <- 0 until order)
                    for (j <- 0 until order) {
                        B(i)(j) += A(j)(i)
                        A(j)(i) += 1.0
                    }
            }
        }

        /*********************************************************************
        ** Analyze and output results.
        *********************************************************************/

        val transposeTime = System.nanoTime() - startTime

        var abserr = 0.0
        val addit = ((iterations+1).toDouble * iterations.toDouble)/2.0
        for (j <- 0 until order) {
            for (i <- 0 until order) {
                abserr += Math.abs(B(j)(i)
                    - ( (order*i + j).toDouble*(iterations+1).toDouble+addit) )
            }
        }

        //println("Sum of absolute differences: " + abserr)

        if (abserr < epsilon) {
            println("Solution validates")
            val avgtime = transposeTime.toDouble*1e-9/iterations.toDouble
            val rate = 1.0e-6 * bytes/avgtime
            println(f"Rate (MB/s): ${rate} Avg time (s): ${avgtime}")
            // println("Squared errors: " + abserr)
        } else {
            println("ERROR: Aggregate squared error ${abserr} exceeds threshold ${epsilon}")
        }
    }
}
