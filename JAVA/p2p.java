public class p2p {
    public static void main (String args[]) {
        System.out.println("Parallel Research Kernels.");
        System.out.println("Java pipeline execution on 2D grid.");


        /*******************************************************************
        **read and test input parameters
        *******************************************************************/

        if (args.length != 3) {
            System.out.println("argument count = " + args.length);
            System.out.println("Usage: java p2p <# iterations> <first array dimension> <second array dimension>");
            return;
        }

        int iterations = Integer.parseInt(args[0]);
        if (iterations < 1) {
            System.out.println("ERROR: iterations must be >= 1");
            return;
        }

        int m = Integer.parseInt(args[1]);
        if (m < 1) {
            System.out.println("ERROR: array dimension must be >= 1");
            return;
        }

        int n = Integer.parseInt(args[2]);
        if (n < 1) {
            System.out.println("ERROR: array dimension must be >= 1");
            return;
        }

        System.out.format("Grid sizes               = %d * %d%n", m, n);
        System.out.format("Number of iterations     = %d%n", iterations);

        double[][] grid = new double[m][n];
        for (int j = 0; j < n; j++) {
            grid[0][j] = j;
        }
        for (int i = 0; i < m; i++) {
            grid[i][0] = i;
        }


        long startTime = 0;
        for (int k = 0; k < iterations+1; k++){
            // start timer after a warmup iteration
            if (k<1)
                startTime = System.currentTimeMillis();

            for (int i = 1; i < m; i++){
                for (int j = 1; j < n; j++){
                    grid[i][j] = grid[i-1][j] + grid[i][j-1] - grid[i-1][j-1];
                }
            }

            // copy top right corner value to bottom left corner to create dependency
            grid[0][0] = -grid[m-1][n-1];
        }

        final long pipelineTime = System.currentTimeMillis() - startTime;

        /*******************************************************************
        **Final analysis
        *******************************************************************/

        double cornerValue = (double) ((iterations+1)*(m+n-2));
        double epsilon = 1.e-8;
        if ( (Math.abs(grid[m-1][n-1] - cornerValue)/cornerValue) < epsilon ) {
            System.out.println("Solution validates");
            double avgtime = pipelineTime/(double)iterations/1000;
            System.out.format("Rate (MFlops/s): %f;  Avg time (s): %f%n", (1.e-6)*2*(m-1)*(n-1)/avgtime,avgtime);
        } else {
            System.out.format("ERROR: checksum %f does not match verification value %f%n",grid[m-1][n-1], cornerValue);
        }


    }
}