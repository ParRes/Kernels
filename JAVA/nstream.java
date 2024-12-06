public class nstream {
    public static void main (String args[]) {
        System.out.println("Parallel Research Kernels.");
        System.out.println("Java Stream triad: A = B + scalar * C.");

        /*******************************************************************
        **read and test input parameters
        *******************************************************************/

        if (args.length != 2){
            System.out.println("Usage: java nstream <# iterations> <vector length>");
            return;
        }

        int iterations = Integer.parseInt(args[0]);
        if (iterations < 1) {
            System.out.println("ERROR: iterations must be >= 1");
            return;
        }

        int length = Integer.parseInt(args[1]);
        if (length < 1) {
            System.out.println("ERROR: vector length must be positive");
            return;
        }

        System.out.println("Vector length        = " + length);
        System.out.println("Number of iterations = " + iterations);

        double[] a = new double[length];
        double[] b = new double[length];
        double[] c = new double[length];

        for (int j = 0; j < length; j++) {
            a[j] = 0.0;
            b[j] = 2.0;
            c[j] = 2.0;
        }

        /* --- MAIN LOOP --- repeat Triad iterations times --- */

        double scalar = 3.0;

        long startTime = 0; // silence compiler warning

        for (int iter=0; iter<=iterations; iter++) {

            /* start timer after a warmup iteration */
            if (iter == 1)
                startTime = System.currentTimeMillis();

            for (int j = 0; j < length; j++)
                a[j] += b[j] + scalar*c[j];
        }

        /*********************************************************************
        ** Analyze and output results.
        *********************************************************************/

        final long streamTime = System.currentTimeMillis() - startTime;

        double ar = 0.0;
        double br = 2.0;
        double cr = 2.0;

        for (int k = 0; k < iterations + 1; k++)
            ar += br + scalar * cr;

        ar *= length;

        double asum = 0.0;
        for (int i = 0; i < length; i++)
            asum += Math.abs(a[i]);


        double epsilon=1.e-8;
        if (Math.abs(ar-asum)/asum > epsilon) {
            System.out.println("Failed Validation on output array");
            System.out.println("        Expected checksum: " + ar);
            System.out.println("        Observed checksum: " + asum);
            System.out.println("ERROR: solution did not validate");
        } else {
            System.out.println("Solution validates");
            double avgtime = streamTime/(double)iterations/1000;
            double nbytes = 4.0 * length * 8;
            System.out.format("Rate (MB/s): %f Avg time (s): %f%n", 1.e-6*nbytes/avgtime, avgtime);
        }
    }
}