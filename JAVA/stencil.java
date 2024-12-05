public class stencil {
    public static void main (String args[]) {
        System.out.println("Parallel Research Kernels.");
        System.out.println("Java pipeline execution on 2D grid.");


        /*******************************************************************
        **read and test input parameters
        *******************************************************************/
        if (args.length < 2 || args.length > 5){
            System.out.println("Usage: java transpose <# iterations> <array dimension> [<star/stencil> <radius> <tile size>]");
            return;
        }

        int iterations = Integer.parseInt(args[0]);
        if (iterations < 1) {
            System.out.println("ERROR: iterations must be >= 1");
            return;
        }

        int n = Integer.parseInt(args[1]);
        if (n < 1) {
            System.out.println("ERROR: grid dimension must be positive: " + n);
            return;
        }

        String pattern = new String();
        if (args.length > 2) {
            pattern = args[2];
        } else {
            pattern = "star";
        }

        int radius;
        if (args.length > 3) {
            radius = Integer.parseInt(args[3]);
            if (radius < 1) {
                System.out.println("ERROR: Stencil radius should be positive." );
                return;
            }
            if ((2*radius + 1) > n) {
                System.out.println("ERROR: Stencil radius exceeds grid size");
                return;
            }
        } else {
            radius = 2;
        }

        int tileSize = 0;
        if (args.length > 4) {
            tileSize = Integer.parseInt(args[4]);
            if (tileSize <= 0 || tileSize>n)
                tileSize=n;
        }

        System.out.println("Number of iterations = " + iterations);
        System.out.println("Grid size            = " + n);

        if (pattern.equals("star")) {
            System.out.println("Type of stencil      = star");
        } else {
            System.out.println("Type of stencil      = stencil");
        }

        if (tileSize != n)
            System.out.println("Tile size            = " + tileSize);
        else
            System.out.println("Untiled");


        System.out.println("radius of stencil    = " + radius);
        System.out.println("Data type            = double precision");
        System.out.println("Compact representation of stencil loop body");

        // intialize the input, output, and weight arrays
        double[][] weight = new double[2*radius + 1][2*radius + 1];
        double[][] in = new double[n][n];
        double[][] out = new double[n][n];

        int stencilSize = 0;
        if (pattern.equals("star")){
            stencilSize = 4*radius + 1;
            for (int i = 1; i < radius + 1; i++){
                weight[radius][radius+i] = +1./(2*i*radius);
                weight[radius+i][radius] = +1./(2*i*radius);
                weight[radius][radius-i] = -1./(2*i*radius);
                weight[radius-i][radius] = -1./(2*i*radius);
            }
        } else {
            stencilSize = (2*radius + 1)*(2*radius + 1);
            for (int j = 1; j < radius + 1; j++) {
                for (int i = -j + 1; i < j; i++) {
                    weight[radius+i][radius+j] = +1./(4*j*(2*j-1)*radius);
                    weight[radius+i][radius-j] = -1./(4*j*(2*j-1)*radius);
                    weight[radius+j][radius+i] = +1./(4*j*(2*j-1)*radius);
                    weight[radius-j][radius+i] = -1./(4*j*(2*j-1)*radius);
                }

                weight[radius+j][radius+j]    = +1./(4*j*radius);
                weight[radius-j][radius-j]    = -1./(4*j*radius);
            }
        }

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++)
                in[i][j] = i+j;
        }

        long startTime = 0;
        double normal = 0.0;
        double activePoints = (n-2*radius)*(n-2*radius);
        double epsilon = 1.e-8;
        int referenceNorm = 2*(iterations + 1);

        for (int iter = 0; iter<=iterations; iter++){

            // start timer after a warmup iteration
            if (iter == 1)
                startTime = System.currentTimeMillis();

            // Apply the stencil operator

            if (tileSize == 0) {
                for (int j = radius; j < n - radius; j++) {
                    for (int i = radius; i< n - radius; i++) {
                        if (pattern.equals("star")) {
                            for (int jj =  -radius; jj <= radius; jj++)
                                out[j][i] += weight[0+radius][jj+radius]*in[j+jj][i];
                            for (int ii = -radius; ii < 0; ii++)
                                out[j][i] += weight[ii+radius][0+radius]*in[j][i+ii];
                            for (int ii = 1; ii <= radius; ii++)
                                out[j][i] += weight[ii+radius][0+radius]*in[j][i+ii];
                        } else {
                            for (int jj =- radius; jj <= radius; jj++) {
                                for (int ii =- radius; ii <= radius; ii++)
                                    out[j][i] += weight[ii+radius][jj+radius]*in[j+jj][i+ii];
                            }
                        }
                    }
                }
            } else {
                for (int jt = radius; jt < n - radius; jt += tileSize) {
                    for (int it = radius; it < n - radius; it += tileSize) {
                        for (int j = jt; j < Math.min(n-radius,jt+tileSize); j++) {
                            for (int i = it; i < Math.min(n-radius,it+tileSize); i++) {
                                if (pattern.equals("star")){
                                    for (int jj =  -radius; jj <= radius; jj++)
                                        out[j][i] += weight[0+radius][jj+radius]*in[j+jj][i];
                                    for (int ii = -radius; ii < 0; ii++)
                                        out[j][i] += weight[ii+radius][0+radius]*in[j][i+ii];
                                    for (int ii = 1; ii <= radius; ii++)
                                        out[j][i] += weight[ii+radius][0+radius]*in[j][i+ii];
                                } else {
                                    for (int jj =- radius; jj <= radius; jj++) {
                                        for (int ii =- radius; ii <= radius; ii++)
                                            out[j][i] += weight[ii+radius][jj+radius]*in[j+jj][i+ii];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // add constant to solution to force refresh of neighbor data, if any
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++)
                    in[j][i] += 1.0;
            }

        }

        /*********************************************************************
        ** Analyze and output results.
        *********************************************************************/

        final long stencilTime = System.currentTimeMillis() - startTime;

        for (int j = radius; j < n - radius; j++) {
            for (int i = radius; i < n - radius; i++) {
                normal += Math.abs(out[j][i]);
            }
        }

        normal /= activePoints;

        if (Math.abs(normal - referenceNorm) < epsilon) {
            System.out.println("Solution validates");
            double flops = (2*stencilSize + 1) * activePoints;
            double avgtime = stencilTime/(double)iterations/1000;
            System.out.format("Rate (MFlops/s): %f Avg time (s):%f%n", 1.e-6*flops/avgtime, avgtime);
        }
        else {
            System.out.format("ERROR: L1 norm = %.10f Reference L1 norm = %d%n", normal, referenceNorm);
        }


    }
}
