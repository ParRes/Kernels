public class transpose {
    public static void main (String args[]) {
        System.out.println("Parallel Research Kernels.");
        System.out.println("Java Matrix transpose: B = A^T.");

        /*******************************************************************
        **read and test input parameters
        *******************************************************************/

        if (args.length != 3 && args.length != 2){
            System.out.println("Usage: java transpose <# iterations> <matrix order> [tile size]");
            return;
        }

        int iterations = Integer.parseInt(args[0]);
        if (iterations < 1) {
            System.out.println("ERROR: iterations must be >= 1");
            return;
        }

        int order = Integer.parseInt(args[1]);
        if (order < 1) {
            System.out.println("ERROR: matrix order must be >= 1");
            return;
        }

        int tileSize = 0;
        if (args.length == 3)
            tileSize = Integer.parseInt(args[2]);
        /* a non-positive tile size means no tiling of the local transpose */
        if (tileSize <=0)
            tileSize = order;

        System.out.println("Matrix order          = " + order);
        if (tileSize < order)
            System.out.println("Tile size             = " + tileSize);
        else
            System.out.println("Untiled");
        System.out.println("Number of iterations  = " + iterations);



        double A[][] = new double[order][order];
        double B[][] = new double[order][order];

        // Fill the original matrix
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                A[i][j] = i*order+j;
            }
        }


        long startTime = 0;
        double epsilon=1.e-8;
        double bytes = 2.0 * 8 * order * order;

        for (int iter = 0; iter <= iterations; iter++){

            /* start timer after a warmup iteration                                        */
            if (iter == 1)
                startTime = System.currentTimeMillis();

            /* Transpose the  matrix; only use tiling if the tile size is smaller
               than the matrix */
            if (tileSize < order) {
                for (int i = 0; i < order; i += tileSize)
                    for (int j = 0; j < order; j += tileSize)
                        for (int it = i; it < Math.min(order,i+tileSize); it++)
                            for (int jt = j; jt < Math.min(order,j+tileSize); jt++){
                                B[it][jt] += A[jt][it];
                                A[jt][it] += 1.0;
                            }
            }
            else {
                for (int i = 0; i < order; i++)
                    for (int j = 0; j < order;j++) {
                        B[i][j] += A[j][i];
                        A[j][i] += 1.0;
                    }
            }
        }

        /*********************************************************************
        ** Analyze and output results.
        *********************************************************************/

        final long transposeTime = System.currentTimeMillis() - startTime;

        double abserr = 0.0;
        double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
        for (int j = 0; j < order; j++) {
            for (int i = 0; i < order; i++) {
                abserr += Math.abs(B[j][i]
                    - ( (double)(order*i + j)*(iterations+1)+addit) );
            }
        }

        //System.out.println("Sum of absolute differences: " + abserr);

        if (abserr < epsilon) {
            System.out.println("Solution validates");
            double avgtime = transposeTime/(double)iterations/1000;
            System.out.format("Rate (MB/s): %f Avg time (s): %f%n",
                1.0e-6 * bytes/avgtime, avgtime);

            //System.out.println("Squared errors: " + abserr);
        } else {
            System.out.format("ERROR: Aggregate squared error %f exceeds threshold %f%n",
               abserr, epsilon);
        }
    }
}
