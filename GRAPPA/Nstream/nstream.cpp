/**********************************************************************
 
NAME:      nstream
 
PURPOSE:   To compute memory bandwidth when adding a vector of a given
           number of double precision values to the scalar multiple of 
           another vector of the same length, and storing the result in
           a third vector. 
 
USAGE:     The program takes as input the number of iterations to loop
           over the triad vectors, the length of the vectors, and the
           offset between vectors.
 
           <progname> <# iterations> <vector length> <offset>
           
           The output consists of diagnostics to make sure the 
           algorithm worked, and of timing statistics.
 
FUNCTIONS CALLED:
 
           Other than OpenMP or standard C functions, the following 
           external functions are used in this program:
 
           wtime()
           bail_out()
           checkTRIADresults()
 
NOTES:     Bandwidth is determined as the number of words read, plus the 
           number of words written, times the size of the words, divided 
           by the execution time. For a vector length of N, the total 
           number of words read and written is 4*N*sizeof(double).
 
HISTORY:   This code is loosely based on the Stream benchmark by John
           McCalpin, but does not follow all the Stream rules. Hence,
           reported results should not be associated with Stream in
           external publications
**********************************************************************/

#include <par-res-kern_general.h>
#include <Grappa.hpp>
#include <cstdint>

#define SCALAR 3.0
#define root 0

static int checkTRIADresults(int32_t, int64_t, GlobalAddress<double> a);

using namespace Grappa;

struct Checksum {
  double asum;
  void init(double asum) {
    this->asum = asum;
  }
} GRAPPA_BLOCK_ALIGNED;


int main(int argc, char *argv[]) {
  Grappa::init( &argc, &argv );

  int my_ID = Grappa::mycore();
  int Num_procs = Grappa::cores();

  if (argc != 3) {
    if (my_ID == root)
      std::cout << "Usage: " << argv[0]
		<< " <# iterations> <vector length> " << std::endl;
    exit(1);
  }

  int32_t iterations = atoi(argv[1]);
  if (iterations < 1) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid number of iterations: " << iterations << std::endl;
    exit(1);
  }

  int64_t length = atol(argv[2]);
  if (length < Num_procs) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid vector length: " << length << std::endl;
    exit(1);
  }

  Grappa::run([=] {

      GlobalAddress<double> a = global_alloc<double>(length*sizeof(double));
      double * abase = a.pointer();
      GlobalAddress<double> b = global_alloc<double>(length*sizeof(double));
      double * bbase = b.pointer();
      GlobalAddress<double> c = global_alloc<double>(length*sizeof(double));
      double * cbase = c.pointer();
      if (!a || !b || !c) {
	std::cout << "ERROR: Could not allocate " << 3*length
		  << " words for vectors." << std::endl;
	exit(1);
      }

      double bytes = 4.0 * sizeof(double) * length;

      std::cout << "Grappa stream triad: A = B + scalar*C" << std::endl;
      std::cout << "Number of threads    = " << Num_procs << std::endl;
      std::cout << "Vector length        = " << length << std::endl;
      // std::cout << "Offset               = " << offset << std::endl;
      std::cout << "Number of iterations = " << iterations << std::endl;
      
      double scalar = SCALAR;
      Grappa::memset(a, 0, length);
      Grappa::memset(b, 2.0, length);
      Grappa::memset(c, 2.0, length);

      double nstream_time = Grappa::walltime();

      // a[j] = b[j] + scalar*c[j];
      for (int iter = 0; iter < iterations; iter++) {
	 // forall(c, length, [=](int64_t i, double& ci) {
	 //    delegate::call<async>(b+i, [=] (double& bi) {
	 // 	delegate::call<async>(a+i, [=] (double& ai) {
	 // 	    ai += bi + SCALAR*ci;
	 // 	  });
	 //      });
	 //  });
	// forall(a, length, [=](int64_t i, double& aref) {
	//     GlobalAddress<double> bi = b + i;
	//     double * bip = bi.pointer();
	    
	//     GlobalAddress<double> ci = c + i;
	//     double * cip = ci.pointer();
	    
	//     aref += *bip + SCALAR * (*cip);
	//   });
	
	forall(a, length, [=](double& aref) {
	    aref += bbase[&aref-abase] + SCALAR * cbase[&aref-abase];
	  });
      }   // end iterations
      nstream_time = Grappa::walltime() - nstream_time;

      /*********************************************************************
       ** Analyze and output results.
       *********************************************************************/
      
      if (checkTRIADresults(iterations, length, a)) {
	double avgtime = nstream_time/iterations;
	std::cout << "Rate (MB/s): " << 1.0E-06*(bytes/avgtime)
		  << " Avg time (s): " << avgtime << std::endl;
      } else {
	std::cout << "You messed up. Exiting..." << std::endl;
      }

      global_free(a);
      global_free(b);
      global_free(c);
    });


    Grappa::finalize();
}


int checkTRIADresults(int32_t iterations, int64_t length, GlobalAddress<double> a) {
  double ak, bk, ck, scalar;
  double epsilon = 1.e-8;
  int j, iter;

  /* reproduce ititialization */
  ak = 0.0;
  bk = 2.0;
  ck = 2.0;

  /* now execute timing loop */
  scalar = SCALAR;
  for (iter = 0; iter < iterations; iter++) {
    ak += bk + scalar * ck;
  }

  ak = ak * (double)length;

  auto cs = symmetric_global_alloc<Checksum>();
  Grappa::on_all_cores([cs] {cs->init(0.0);});
  forall (a, length, [=] (double& aj) {
      on_all_cores([cs, aj] {cs->asum += aj;});
    });

  if (ABS(ak-cs->asum)/cs->asum > epsilon) {
    std::cout << "Failed Validation on output array" << std::endl;
    std::cout << "Expected checksum: " << ak << std::endl;
    std::cout << "Observed checksum: " << cs->asum << std::endl;
    return 0;
  } else {
    std::cout << "Solution validates" << std::endl;
    return 1;
  }
}
