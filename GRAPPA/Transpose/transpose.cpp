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

NAME:    transpose

PURPOSE: This program tests the efficiency with which a square matrix
         can be transposed and stored in another matrix. The matrices
         are distributed identically.
  
USAGE:   Program inputs are the matrix order, the number of times to 
         repeat the operation, and the communication mode

         transpose <# iterations> <matrix order> [tile size]

         An optional parameter specifies the tile size used to divide the 
         individual matrix blocks for improved cache and TLB performance. 
  
         The output consists of diagnostics to make sure the 
         transpose worked and timing statistics.

FUNCTIONS CALLED:

         Other than MPI or standard C functions, the following 
         functions are used in this program:

          wtime()           Portable wall-timer interface.
          bail_out()        Determine global error and exit if nonzero.

HISTORY: Written by Tim Mattson, April 1999.  
         Updated by Rob Van der Wijngaart, December 2005.
         Updated by Rob Van der Wijngaart, October 2006.
         Updated by Rob Van der Wijngaart, November 2014::
         - made variable names more consistent 
         - put timing around entire iterative loop of transposes
         - fixed incorrect matrix block access; no separate function
           for local transpose of matrix block
         - reordered initialization and verification loops to
           produce unit stride
         - changed initialization values, such that the input matrix
           elements are: A(i,j) = i+order*j
         
  
*******************************************************************/

/******************************************************************
                     Layout nomenclature                         
                     -------------------

o Each rank owns one block of columns (Colblock) of the overall
  matrix to be transposed, as well as of the transposed matrix.
o Colblock is stored contiguously in the memory of the rank. 
  The stored format is column major, which means that matrix
  elements (i,j) and (i+1,j) are adjacent, and (i,j) and (i,j+1)
  are "order" words apart
o Colblock is logically composed of #ranks Blocks, but a Block is
  not stored contiguously in memory. Conceptually, the Block is 
  the unit of data that gets communicated between ranks. Block i of 
  rank j is locally transposed and gathered into a buffer called Work, 
  which is sent to rank i, where it is scattered into Block j of the 
  transposed matrix.
o When tiling is applied to reduce TLB misses, each block gets 
  accessed by tiles. 
o The original and transposed matrices are called A and B

 -----------------------------------------------------------------
|           |           |           |                             |
| Colblock  |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |  Block    |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |Tile|      |           |                             |
|           |    |      |           |   Overall Matrix            |
|           |----       |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
 -----------------------------------------------------------------*/

#include <par-res-kern_general.h>
#include <Grappa.hpp>
#include <FullEmpty.hpp>

using namespace Grappa;

struct Timer {
  double start;
  double total;
} GRAPPA_BLOCK_ALIGNED;
struct AbsErr {
  double abserr;
} GRAPPA_BLOCK_ALIGNED;

#define A(i,j)        A_p[(i+istart)+order*(j)]
#define B(i,j)        B_p[(i+istart)+order*(j)]
#define Work_out(i,j) Work_out_p[i+Block_order*(j)]

#define root 0

#define symmetric static

const int CHUNK_LENGTH = 16;
typedef double row_t[CHUNK_LENGTH];

int main(int argc, char * argv[]) {
  Grappa::init( &argc, &argv );

  int Num_procs = Grappa::cores();

  if (argc != 3 && argc != 4) {
    if (Grappa::mycore() == root)
      std::cout << "Usage: " << argv[0] << " <#iterations> <matrix order> [tile size]"
		<< std::endl;
    exit(1);
  }

  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    if (Grappa::mycore() == root) 
      std::cout << "ERROR: iterations must be >= 1 : " << iterations << std::endl;
    exit(1);
  }

  int order = atoi(argv[2]);
  if (order < 0) {
    if (Grappa::mycore() == root) 
      std::cout << "ERROR: Matrix Order must be greater than 0 : " << order << std::endl;
    exit(1);
  }
  if (order%Num_procs) {
    if (Grappa::mycore() == root)
      std::cout << "ERROR: order "<<order<<" should be divisible by # of procs "
		<<Num_procs<<std::endl;
    exit(1);
  }

  Grappa::run( [iterations, order, Num_procs, argc, argv] {
      symmetric int my_ID;
      symmetric int64_t Block_order,Block_size,Colblock_size,colstart;
      symmetric double * A_p, * B_p;
      symmetric double * Work_out_p;
      symmetric double abserr, epsilon = 1.e-8;
      double trans_time, avgtime, abserr_tot;

      int Tile_order = 32; /* default tile size for tiling of local transpose */
      if (argc == 4) Tile_order = atoi(argv[3]);
      if (Tile_order%CHUNK_LENGTH != 0){
	std::cout<<"ERROR: Tile Order: "<<Tile_order
		 <<" must be a multiple of const CHUNK_LENGTH "
		 <<CHUNK_LENGTH<<std::endl;
	exit(1);
      }

      int64_t tiling = (Tile_order > 0) && (Tile_order < order);
      if (!tiling) Tile_order = order;
      int64_t bytes = 2.0 * sizeof(double) * order * order;

      Grappa::on_all_cores( [=] {
	  my_ID = Grappa::mycore();
	  int64_t i, j, istart; // dummies 

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a 
** rank.  Each column block is made up of Num_procs smaller square 
** blocks of order block_order.
*********************************************************************/
	  
	  Block_order   = order / Num_procs;
	  colstart      = Block_order * my_ID;
	  Colblock_size = order * Block_order;
	  Block_size    = Block_order * Block_order;


/*********************************************************************
** Create the column block of the test matrix, the row block of the 
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
	  A_p = Grappa::locale_new_array<double>(Colblock_size);
	  B_p = Grappa::locale_new_array<double>(Colblock_size);
	  if (!A_p || !B_p) {
	    std::cout << "Core " << my_ID << " could not allocate space for "
		      << "matrices" << std::endl;
	    exit(1);
	  }
	  if (Num_procs > 1) {
	    Work_out_p = Grappa::locale_new_array<double>(Block_size);
	    if (!Work_out_p){
	      std::cout << "Error allocating space for work on node " << my_ID << std::endl;
	      exit(1);
	    }
	  }

	  // fill the original column matrix
	  istart = 0;
	  for (j = 0; j < Block_order; j++)
	    for (i = 0; i < order; i++) {
	      A(i,j) = (double)(order * (j+colstart) + i);
	      B(i,j) = -1.0;
	    }

	});

      // TODO: get rid of this restriction
      tiling = tiling && (Block_order%CHUNK_LENGTH == 0);

      std::cout << "Grappa matrix transpose: B = A^T" << std::endl;
      std::cout << "Number of cores         = " << Num_procs << std::endl;
      std::cout << "Matrix order            = " << order << std::endl;
      std::cout << "Number of iterations    = " << iterations << std::endl;
      if (tiling) std::cout << "Tile size               = " << Tile_order << std::endl;
      else std::cout << "Untiled" << std::endl;

      GlobalAddress<Timer> timer = Grappa::symmetric_global_alloc<Timer>();

      Grappa::finish( [=] {
      Grappa::on_all_cores( [=] {
	  int send_to, recv_from;
	  int64_t i, j, it, jt, istart, iter, phase; // dummies 
	  double val;
	  int target;

	  for ( iter = 0; iter < iterations; iter++) {
	    
	    // start timer after warmup iteration
	    if (iter == 1) timer->start = Grappa::walltime();

	  // execute kernel

	    // do the local transpose
	    istart = colstart;
	    if (!tiling) {
	      for (i=0; i<Block_order; i++)
		for (j=0; j<Block_order; j++) {
		  B(j,i) = A(i,j);
		}
	    } else {
	      for (i=0; i<Block_order; i+=Tile_order) 
		for (j=0; j<Block_order; j+=Tile_order) 
		  for (it=i; it<MIN(Block_order,i+Tile_order); it++)
		    for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++)
		      B(jt,it) = A(it,jt); 
	    }

	    for (phase=1; phase<Num_procs; phase++) {
	      
	      recv_from = (my_ID + phase             )%Num_procs;
	      send_to   = (my_ID - phase + Num_procs  )%Num_procs;
	      
	      istart = send_to*Block_order;
	      if (!tiling) {
		for (i=0; i<Block_order; i++) 
		  for (j=0; j<Block_order; j++){
		    Work_out(j,i) = A(i,j);
		  }
	      }
	      else {
		for (i=0; i<Block_order; i+=Tile_order) 
		  for (j=0; j<Block_order; j+=Tile_order) 
		    for (it=i; it<MIN(Block_order,i+Tile_order); it++)
		      for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
			Work_out(jt,it) = A(it,jt);
		      }
	      }

	      istart = my_ID * Block_order;
	      // write local buffer to transposed matrix
	      if (!tiling) {
		for (i=0; i<Block_order; i++) 
		  for (j=0; j<Block_order; j++) {
		    target = i+istart+(order*j);
		    val = Work_out(i,j);
		    Grappa::delegate::call<async>(send_to, [val,target] {
			B_p[target] = val;
		      });
		  }
	      }
	      else {
		for (i=0; i<Block_order; i+=Tile_order) 
		  for (j=0; j<Block_order; j+=Tile_order) 
		    for (it=i; it<MIN(Block_order,i+Tile_order); it+=CHUNK_LENGTH)
		      for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
			target = it+istart+(order*jt);
			row_t& row = *reinterpret_cast<row_t*>(&Work_out(it,jt));
			Grappa::delegate::call<async>(send_to, [row,target] {
			    memcpy(&B_p[target], &row, sizeof(row_t));
			  });
		      }
	      }	       

	      // ensures all async writes complete before moving to next phase
	      Grappa::impl::local_ce.wait();

        barrier();
	    } // end of phase loop
	  } // done with iterations
	});
	});

      Grappa::on_all_cores( [timer] {
	  timer->total = Grappa::walltime() - timer->start;
	});

      GlobalAddress<AbsErr> ae = Grappa::symmetric_global_alloc<AbsErr>();
      // check for errors
      Grappa::on_all_cores( [=] {
	  int64_t i,j;
	  int64_t istart = 0;
	  ae->abserr = 0;
	  for (j=0;j<Block_order;j++) for (i=0;i<order;i++) {
	      ae->abserr += ABS(B(i,j) - (double)(order*i + j+colstart));
	      if (B(i,j) != order*i+j+colstart){
	      	LOG(INFO)<<"Expected: "<<order*i+j+colstart<<"  Observed: "
	      		 <<B(i,j)<<" at ("<<i<<","<<j<<")";
	      }
	    }
	});

      abserr_tot = Grappa::reduce<double, collective_sum<double>>(&ae->abserr);
      trans_time = Grappa::reduce<double,collective_max<double>>( &timer->total );
      if (abserr_tot < epsilon) {
	std::cout << "Solution validates" << std::endl;
	avgtime = trans_time/(double)iterations;
	std::cout << "Rate (MB/s): " << 1.0E-06*bytes/avgtime
		  << " Avg time (s): " << avgtime << std::endl;
	std::cout << "Summed errors: " << abserr_tot << std::endl;
      } else {
	std::cout << "ERROR: Aggregate squared error " << abserr_tot
		  << " exceeds threshold " << epsilon << std::endl;
      }
    });

  Grappa::finalize();
}
