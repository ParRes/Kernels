/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
* Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

/* Copyright 1991-2013: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*           "tuned STREAM benchmark results"                            */
/*           "based on a variant of the STREAM benchmark code"           */
/*         Other comparable, clear, and reasonable labelling is          */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/




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
 
           Other than MPI or standard C functions, the following 
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
REVISION:  Modified by Rob Van der Wijngaart, December 2005, to     
           parameterize vector size and offsets through compiler flags.
           Also removed all Stream cases except TRIAD.                 
REVISION:  Modified by Rob Van der Wijngaart, March 2006, to handle MPI.
REVISION:  Modified by Rob Van der Wijngaart, May 2006, to introduce
           dependence between successive triad operations. This is
           necessary to avoid dead code elimination
REVISION:  Modified by Rob Van der Wijngaart, November 2014, replaced
           timing of individual loop iterations with timing of overall
           loop; also replaced separate loop establishing dependence
           between iterations (must now be included in timing) with
           accumulation: a[] += b[] + scalar*c[]
**********************************************************************/
 
#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>
#include <Grappa.hpp>
#include <cstdint>

 
#define SCALAR  3.0
#define root 0

using namespace Grappa;

static int checkTRIADresults(int, long int, double *);

struct Timer {
  double start;
  double total;
} GRAPPA_BLOCK_ALIGNED;
 
int main(int argc, char **argv) {
  Grappa::init( &argc, &argv );

/**********************************************************************************
* process and test input parameters    
***********************************************************************************/

  int my_ID = Grappa::mycore();
  int Num_procs = Grappa::cores();

  if (argc != 4) {
    if (my_ID == root)
      std::cout << "Usage: " << argv[0] 
		<< " <# iterations> <vector length> <offset>" << std::endl;
    exit(1);
  }

  int iterations   = atoi(argv[1]);
  if (iterations < 1) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid number of iterations: " << iterations << std::endl;
    exit(1);
  }

  int64_t total_length = atol(argv[2]);
  int64_t length;
  if (total_length < Num_procs) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid vector length: " << total_length << std::endl;
    exit(1);
  } else { length = total_length/Num_procs; }

  int64_t offset = atol(argv[3]);
  if (offset < 0) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid array offset: " << offset << std::endl;
    exit(1);
  }

  Grappa::run([Num_procs, iterations, length, offset] {
      static double * a, * b, * c;
      double scalar, nstream_time, avgtime;   
      double bytes   = 4.0 * sizeof(double) * length * Num_procs;

      Grappa::on_all_cores([=] {
	  a = new double[3*length + 2*offset];
	  if (!a) {
	    std::cout << "ERROR: Core " << Grappa::mycore()
		      << "Could not allocate " << 3*length + 2*offset
		      << "bytes for vectors" << std::endl;
	    exit(1);
	  }

	  b = a + length + offset;
	  c = b + length + offset;
	  
	  for (int64_t j=0; j<length; j++) {
	    a[j] = 0.0;
	    b[j] = 2.0;
	    c[j] = 2.0;
	  }
	});

      std::cout << "Grappa stream triad: A = B + scalar*C" << std::endl;
      std::cout << "Number of ranks      = " << Num_procs << std::endl;
      std::cout << "Vector length        = " << length*Num_procs << std::endl;
      std::cout << "Offset               = " << offset << std::endl;
      std::cout << "Number of iterations = " << iterations << std::endl;

      GlobalAddress<Timer> timer = Grappa::symmetric_global_alloc<Timer>();
      scalar = SCALAR;

      /* --- MAIN LOOP --- repeat Triad iterations times --- */
      for (int iter = 0; iter <= iterations; iter++) {
	// allow warmup iteration
	Grappa::on_all_cores([iter, timer] {
	    if (iter == 1) timer->start = Grappa::walltime();
	  });
	
	// execute kernel
	Grappa::on_all_cores([=] {
	    
	    for (int64_t j=0; j<length; j++) a[j] += b[j]+scalar*c[j];
	    
	  });
      }

      Grappa::on_all_cores([timer] {
	  Grappa::barrier();
	  timer->total = Grappa::walltime() - timer->start;
	});
	
      /*********************************************************************
       ** Analyze and output results.
       *********************************************************************/

      if (checkTRIADresults(iterations, length, a)) {
	nstream_time = Grappa::reduce<double, collective_max<double>>( &timer->total );
	avgtime = nstream_time/iterations;
	std::cout << "Rate (MB/s): " <<  1.0E-06 * bytes/avgtime
		  << " Avg time (s): " << avgtime << std::endl;;
      }
      else {
	std::cout << "Solution not valid" << std::endl;
	exit(1);
      }
    });
  Grappa::finalize();
}
 
 
int checkTRIADresults (int iterations, long int length, double *a) {
  double aj, bj, cj, scalar, asum;
  double epsilon = 1.e-8;
  long int j;
  int iter;
 
  /* reproduce initialization */
  aj = 0.0;
  bj = 2.0;
  cj = 2.0;
 
  /* now execute timing loop */
  scalar = SCALAR;
  for (iter=0; iter<=iterations; iter++) aj += bj+scalar*cj;
 
  aj = aj * (double) (length);
 
  asum = 0.0;
  for (j=0; j<length; j++) asum += a[j];
 
#ifdef VERBOSE
  printf ("Results Comparison: \n");
  printf ("        Expected checksum: %f\n",aj);
  printf ("        Observed checksum: %f\n",asum);
#endif
 
  if (ABS(aj-asum)/asum > epsilon) {
    printf ("Failed Validation on output array\n");
#ifndef VERBOSE
    printf ("        Expected checksum: %f \n",aj);
    printf ("        Observed checksum: %f \n",asum);
#endif
    return (0);
  }
  else {
    printf ("Solution validates\n");
    return (1);
  }
}
 
