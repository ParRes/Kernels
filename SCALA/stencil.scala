// based on Java version
object stencil {
    def main (args: Array[String]) : Unit = {
        System.out.println("Parallel Research Kernels.")
        System.out.println("Scala pipeline execution on 2D grid.")

        /*******************************************************************
        **read and test input parameters
         *******************************************************************/
        def error_and_exit(s: String) : Unit = {
            println(s)
            System.exit(1)
        }

        if (args.length < 2 || args.length > 5){
            error_and_exit("Usage: java transpose <# iterations> <array dimension> [<star/stencil> <radius> <tile size>]")
        }

        val iterations = args(0).toInt
        if (iterations < 1) {
            error_and_exit("ERROR: iterations must be >= 1")
        }

        val n = args(1).toInt
        if (n < 1) {
            error_and_exit("ERROR: grid dimension must be positive: " + n)
        }

        val pattern = if (args.length > 2) args(2) else "star"
        val radius = if (args.length > 3) args(3).toInt else 2

        if (radius < 1) {
            error_and_exit("ERROR: Stencil radius should be positive.")
        }
        if ((2*radius + 1) > n) {
            error_and_exit("ERROR: Stencil radius exceeds grid size")
        }

        val tileSize =
            if (args.length > 4) {
                val tmp = args(4).toInt
                if (tmp <= 0 || tmp>n) n else tmp
            } else 0

        println("Number of iterations = " + iterations)
        println("Grid size            = " + n)

        if (pattern == "star") {
            println("Type of stencil      = star")
        } else {
            println("Type of stencil      = stencil")
        }

        if (tileSize != n)
            println("Tile size            = " + tileSize)
        else
            println("Untiled")


        println("radius of stencil    = " + radius)
        println("Data type            = double precision")
        println("Compact representation of stencil loop body")

        // intialize the input, output, and weight arrays
        val weight = Array.ofDim[Double](2*radius + 1, 2*radius + 1)
        val in = Array.ofDim[Double](n, n)
        val out = Array.ofDim[Double](n, n)

        val stencilSize = if (pattern == "star") 4*radius + 1 else (2*radius + 1)*(2*radius + 1)

        if (pattern == "star") {
            for (i <- 1 until radius+1) {
                weight(radius)(radius+i) = +1.0/(2*i*radius)
                weight(radius+i)(radius) = +1.0/(2*i*radius)
                weight(radius)(radius-i) = -1.0/(2*i*radius)
                weight(radius-i)(radius) = -1.0/(2*i*radius)
            }
        } else {
            for (j <- 1 until radius + 1) {
                for (i <- -j + 1 until j) {
                    weight(radius+i)(radius+j) = +1.0/(4*j*(2*j-1)*radius)
                    weight(radius+i)(radius-j) = -1.0/(4*j*(2*j-1)*radius)
                    weight(radius+j)(radius+i) = +1.0/(4*j*(2*j-1)*radius)
                    weight(radius-j)(radius+i) = -1.0/(4*j*(2*j-1)*radius)
                }

                weight(radius+j)(radius+j)    = +1.0/(4*j*radius)
                weight(radius-j)(radius-j)    = -1.0/(4*j*radius)
            }
        }

        for (i <- 0 until n) {
            for (j <- 0 until n)
                in(i)(j) = i + j
        }

        var startTime = System.nanoTime()
        var normal = 0.0
        var activePoints = ((n-2*radius)*(n-2*radius)).toDouble
        var epsilon = 1.0e-8
        var referenceNorm = 2*(iterations + 1)

        for (iter <- 0 to iterations) {

            // start timer after a warmup iteration
            if (iter == 1)
                startTime = System.currentTimeMillis()

            // Apply the stencil operator
            if (tileSize == 0) {
                for (j <- radius until n - radius) {
                    for (i <- radius until n - radius) {
                        if (pattern == "star") {
                            for (jj <- -radius to radius)
                                out(j)(i) += weight(0+radius)(jj+radius)*in(j+jj)(i)
                            for (ii <- -radius until 0)
                                out(j)(i) += weight(ii+radius)(0+radius)*in(j)(i+ii)
                            for (ii <- 1 to radius)
                                out(j)(i) += weight(ii+radius)(0+radius)*in(j)(i+ii)
                        } else {
                            for (jj <- -radius to radius) {
                                for (ii <- -radius to radius)
                                    out(j)(i) += weight(ii+radius)(jj+radius)*in(j+jj)(i+ii)
                            }
                        }
                    }
                }
            } else {
                for (jt <- radius until n - radius by tileSize) {
                    for (it <- radius until n - radius by tileSize) {
                        for (j <- jt until Math.min(n-radius,jt+tileSize)) {
                            for (i <- it until Math.min(n-radius,it+tileSize)) {
                                if (pattern == "star") {
                                    for (jj <-  -radius to radius)
                                        out(j)(i) += weight(0+radius)(jj+radius)*in(j+jj)(i)
                                    for (ii <- -radius until 0)
                                        out(j)(i) += weight(ii+radius)(0+radius)*in(j)(i+ii)
                                    for (ii <- 1 to radius)
                                        out(j)(i) += weight(ii+radius)(0+radius)*in(j)(i+ii)
                                } else {
                                    for (jj <- -radius to radius) {
                                        for (ii <- -radius to radius)
                                            out(j)(i) += weight(ii+radius)(jj+radius)*in(j+jj)(i+ii)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // add constant to solution to force refresh of neighbor data, if any
            for (j <- 0 until n) {
                for (i <- 0 until n)
                    in(j)(i) += 1.0
            }
        }

        /*********************************************************************
        ** Analyze and output results.
        *********************************************************************/

        val stencilTime = System.currentTimeMillis() - startTime

        for (j <- radius until n - radius) {
            for (i <- radius until n - radius) {
                normal += Math.abs(out(j)(i))
            }
        }

        normal /= activePoints

        if (Math.abs(normal - referenceNorm) < epsilon) {
            println("Solution validates")
            val flops = ((2*stencilSize + 1) * activePoints).toDouble
            val avgtime = stencilTime/iterations.toDouble/1000.0
            val mflops = 1.0e-6*flops/avgtime
            println(f"Rate (MFlops/s): ${mflops} Avg time (s): ${avgtime}")
        }
        else {
            println("ERROR: L1 norm = ${normal}%.10f Reference L1 norm = ${referenceNorm}")
        }
    }
}
