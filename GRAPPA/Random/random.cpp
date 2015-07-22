/*
Copyright (c) 2013, Intel Corporation
Copyright (c) 2015, John Abercrombie

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

/******************************************************************* 
 
NAME:    RandomAccess

PURPOSE: This program tests the efficiency of the memory subsystem to 
         update elements of an array with irregular stride.

USAGE:   The program takes as input the number of threads involved, the 2log 
         of the size of the table that gets updated, the ratio of table size 
         over number of updates, and the vector length of simultaneously
         updatable table elements. When multiple threads participate, they
         all share the same table. This can lead to conflicts in memory 
         accesses. Setting the ATOMICFLAG variable in the Makefile will
         avoid conflicts, but at a large price, because the atomic
         directive is currently not implemented on IA for the type of update
         operation used here. Instead, a critical section is used, serializing
         the main loop.
         If the CHUNKFLAG variable is set, contiguous, non-overlapping chunks
         of the array are assigned to individual threads. Each thread computes
         all pseudo-random indices into the table, but only updates table
         elements that fall inside its chunk. Hence, this version is safe, and
         there is no false sharing. It is also non-scalable.

         <progname>  <log2 tablesize> <#update ratio> <vector length>

FUNCTIONS CALLED:

         Other than GRAPPA or standard C functions, the following 
         functions are used in this program:

         wtime()
         PRK_starts()
         poweroftwo()

NOTES:   This program is derived from HPC Challenge Random Access. The random 
         number generator computes successive powers of 0x2, modulo the 
         primitive polynomial x^63+x^2+x+1. The principal differences between 
         this code and the HPCC version are:
         - we start the stream of random numbers not with seed 0x1, but the 
           SEQSEED-th element in the stream of powers of 0x2.
         - the timed code applies the RandomAccess operator twice to the table of 
           computed resuls (starting with the same seed(s) for "ran" in both 
           iterations. The second pass makes sure that any update to any table 
           element that sets high-order bits in the first pass resets those bits 
           to zero.
         - the verification test now simply constitutes checking whether table
           element j equals j.
         - the number of independent streams (vector length) can be changed by the 
           user.

         We note that the vectorized version of this code (i.e. nstarts unequal 
         to 1), does not feature exactly the same sequence of accesses and 
         intermediate update values in the table as the scalar version. The 
         reason for this is twofold.
         1. the elements in the stream of powers of 0x2 that get computed outside 
            the main timed loop as seeds for independent streams in the vectorized 
            version, using the jump-ahead function PRK_starts, are computed inside 
            the timed loop for the scalar version. However, since both versions do 
            the same number of timed random accesses, the vectorized version must
            progress further in the sequence of powers of 0x2.
         2. The independent streams of powers of 0x2 computed in the vectorized 
            version can (and will) cause updates of the same elements of the table 
            in an order that is not consistent with the scalar version. That is, 
            distinct values of "ran" can refer to the same table element 
            "ran&(tablesize-1)," but the operation 
            Table[ran&(tablesize-1)] ^= ran will deposit different values in that 
            table element for different values of ran. At the end of each pass over 
            the data set, the table will contain the same values in the vector and 
            scalar version (ignoring the small differences caused by 1.) because of 
            commutativity of the XOR operator. If the update operator had been 
            non-commutative, the vector and scalar version would have yielded 
            different results.

HISTORY: Written by Rob Van der Wijngaart, June 2006.
         Histogram code (verbose mode) courtesy Roger Golliver
         Shared table version derived from random.c by Michael Frumkin, October 2006
  
************************************************************************************/

#include <par-res-kern_general.h>
#include <Grappa.hpp>
#include <cstdint>

/* Define constants                                                                */
/* PERIOD = (2^63-1)/7 = 7*73*127*337*92737*649657                                 */
#ifdef LONG_IS_64BITS 
  #define POLY               0x0000000000000007UL
  #define PERIOD             1317624576693539401L
  /* sequence number in stream of random numbers to be used as initial value       */
  #define SEQSEED            834568137686317453L
#else 
  #define POLY               0x0000000000000007ULL
  #define PERIOD             1317624576693539401LL
  /* sequence number in stream of random numbers to be used as initial value       */
  #define SEQSEED            834568137686317453LL
#endif
#define ERRORPERCENT 0
#define root 0 


uint64_t PRK_starts(uint64_t n);
static int32_t poweroftwo(int32_t);

struct error {
  int64_t count;
  void init(int64_t count) {
    this->count = count;
  }
} GRAPPA_BLOCK_ALIGNED;

using namespace Grappa;

// define custom statistics which are logged by the runtime
// (here we're not using these features, just printing them ourselves)
GRAPPA_DEFINE_METRIC( SimpleMetric<double>, random_time, 0.0 );

int main(int argc, char * argv[]) {
  /*
    my_ID                = Grappa root core
    Num_procs            = number of cores
    update_ratio         = multiplier of tablesize for # updates
    nstarts              = vector length
    nupdate              = number of updates per thread
    tablesize            = aggregate table size
    log2tablesize        = log2 of aggregate table size
    log2nproc            = log2 of number of procs
    log2update_ratio     = log2 of update ratio
    tablespace           = bytes required for table  
    i,oldsize            = dummies
    ran                  = vector of random numbers
    index                = index into the table
    e                    = error counting struct
   */

  init( &argc, &argv );
  
  int32_t my_ID = Grappa::mycore();
  int32_t Num_procs = Grappa::cores();
  

  if (argc != 4) {
    if (my_ID == root)
      std::cout << "Usage: " << argv[0]
		<< "\n<log2 tablesize> <#update ratio> <vector length>" 
		<< std::endl;
    exit(1);
  }

  // number of procs must be a power of two
  int32_t log2nproc = poweroftwo(Num_procs);
  if (log2nproc < 0) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid number of procs: " << Num_procs 
		<< ", must be a power of 2" << std::endl;
    exit(1);
  }

  int64_t log2tablesize = atoi(argv[1]);
  if (log2tablesize < 1) {
    if (my_ID == root)
      std::cout << "ERROR: Log2 tablesize is " << log2tablesize 
		<< "; must be >= 1" << std::endl;
    exit(1);
  }

  // update ratio must be a power of two
  int32_t update_ratio = atoi(argv[2]);
  int32_t log2update_ratio = poweroftwo(update_ratio);
  if (log2update_ratio < 0) {
    if (my_ID == root)
      std::cout << "ERROR: Invalid update ratio: " << update_ratio
		<< ", must be a power of 2" << std::endl;
    exit(1);
  }

  int32_t nstarts = atoi(argv[3]);
  // vector length must be positive
  if (nstarts < 1) {
    if (my_ID == root)
      std::cout << "ERROR: vector length " << nstarts
		<< ", must be positive" << std::endl;
    exit(1);
  }

  // additional divisibility tests
  if (nstarts % Num_procs) {
    if (my_ID == root)
      std::cout << "ERROR: vector length " << nstarts << " must be divisible by "
		<< "# threads " << Num_procs << std::endl;
    exit(1);
  }
  
  // compute table size carefully to make sure it can be represented
  int64_t tablesize = 1;
  int64_t oldsize;
  for (int i = 0; i < log2tablesize; i++) {
    oldsize = tablesize;
    tablesize <<= 1;
    if (tablesize/2 != oldsize) {
      std::cout << "Requested table size too large; reduce log2 tablesize = "
		<< log2tablesize << std::endl;
      exit(1);
    }
  }

  // even though the table size can be represented, computing the space
  // requred for the table may lead to overflow
  int64_t tablespace = (size_t) tablesize*sizeof(uint64_t);
  if ((tablespace/sizeof(uint64_t)) != tablesize || tablespace <= 0) {
    std::cout << "Cannot represent space for table on this system; "
	      << "reduce log2 tablesize" << std::endl;
    exit(1);
  }

  // compute number of updates carefully to make sure it can be represented
  uint64_t nupdate = update_ratio * tablesize;
  if (nupdate/tablesize != update_ratio) {
    std::cout << "Requested number of updates too large; reduce log2 tablesize "
	      << "or update ratio" << std::endl;
    exit(1);
  }

  Grappa::run([=]{

      std::cout << "Grappa Random Access test" << std::endl;
      std::cout << "Number of threads = " << Num_procs << std::endl;
      std::cout << "Table size (shared) = " << tablesize << std::endl;
      std::cout << "Update ratio = " << update_ratio << std::endl;
      std::cout << "Number of updates = " << nupdate << std::endl;
      std::cout << "Vector length = " << nstarts << std::endl;
      std::cout << "Percent errors allowed = " << ERRORPERCENT << std::endl;

      // create target array that we'll be updating
      auto Table = global_alloc<int64_t>(tablespace);
      forall( Table, tablespace, [=] (int64_t index, int64_t& n) {
	n = index;
	});

      
      // create array of random indices
      int64_t ranspace = nstarts*sizeof(uint64_t);
      auto ran = global_alloc<int64_t>(ranspace);
      
      double start = walltime();
      Metrics::start_tracing();
      
      // do two identical rounds of Random Access to make sure we recover
      // the initial condition
      for (int round = 0; round < 2; round++) {
	forall( ran, nstarts, [=](int64_t index, int64_t& s) {
	    s = PRK_starts(SEQSEED+(nupdate/nstarts)*index);
	  });
	
	forall(ran, nstarts, [=](int64_t& n) {
	    // because we do two rounds, we divide nupdates in two
	    for (int j = 0; j < nupdate/(nstarts*2); j++) {
	      n = (n << 1) ^ (n < 0? POLY : 0);
	      int64_t rand = n & (tablesize-1);
	      delegate::call<async> (Table+rand, [=] (int64_t & t){ t ^= rand;});
	    }
	  });
      }

      random_time = walltime() - start;
      
      Metrics::stop_tracing();
      
      // verification test
      GlobalAddress<error> e = symmetric_global_alloc<error>();
      on_all_cores([e] {e->init(0);});
      forall (Table, tablesize, [=] (int64_t index, int64_t& s) {
	  if (index != s) on_all_cores([e]{e->count++;});
	    });

      if (e->count && (ERRORPERCENT == 0)) {
	if(my_ID == root)
	  std::cout << "ERROR: number of incorrect table elements = "
		    << e->count << std::endl;
	exit(1);
      } else {
	std::cout << "Solution validates, number of errors: " << e->count << std::endl;
	std::cout << "Rate (GUPs/s): " << 1.e-9*nupdate/random_time.value()
		  << ", time (s) = " << random_time.value() << " seconds" << std::endl;
      }

      global_free(Table);
      global_free(ran);
      
    });
  finalize();
}


// Utility routine to start random number generator at nth step
uint64_t PRK_starts(uint64_t n) {

  int i, j;
  uint64_t m2[64];
  uint64_t temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) { 
    temp = 0; 
    for (j=0; j<64; j++)
      if ((uint64_t)((ran >> j) & 1)) 
        temp ^= m2[j]; 
    ran = temp; 
    i -= 1; 
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY : 0); 
  } 
  
  return ran; 
}

int32_t poweroftwo(int32_t n) {
  int32_t log2n = 0;

  while ((1 << log2n) < n) log2n++;
  if (1 << log2n != n) return -1;
  else return log2n;
}
