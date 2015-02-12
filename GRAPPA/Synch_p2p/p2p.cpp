////////////////////////////////////////////////////////////////////////
// TO BE REPLACED BY THE RIGHT LICENSE FILE
////////////////////////////////////////////////////////////////////////

#include <Grappa.hpp>
#include <GlobalAllocator.hpp>
#include <FullEmpty.hpp>
#include <array/GlobalArray.hpp>

using namespace Grappa;

DEFINE_uint64(m, 128, "number of rows in array");
DEFINE_uint64(n, 128, "number of columns in array");
DEFINE_uint64(iterations, 100, "number of iterations");
DEFINE_string(pattern, "border", "what pattern of kernel should we run?");

#define ARRAY(i,j) (local[(j)+((i)*dim2_percore)])

double * local = NULL;
int dim1_size = 0;
int dim2_size = 0;
int dim2_percore = 0;

GlobalArray< double, int, Distribution::Local, Distribution::Block > ga;
GlobalArray< FullEmpty< double >, int, Distribution::Local, Distribution::Block > leftsa;
FullEmpty<double> * lefts = NULL;

struct Timer {
  double start;
  double total;
} GRAPPA_BLOCK_ALIGNED;

int main( int argc, char * argv[] ) {
  init( &argc, &argv );
  run([]{
      LOG(INFO) << "Grappa pipeline stencil execution on 2D ("
                << FLAGS_m << "x" << FLAGS_n
                << ") grid";
      
      ga.allocate( FLAGS_m, FLAGS_n );
      leftsa.allocate( FLAGS_m, Grappa::cores() );
      on_all_cores( [] {
          lefts = new FullEmpty<double>[FLAGS_n];
        } );

      double avgtime = 0.0;
      double mintime = 366.0*24.0*3600.0;
      double maxtime = 0.0;

      // initialize
      LOG(INFO) << "Initializing....";
      forall( ga, [] (int i, int j, double& d) {
          if( i == 0 ) {
            d = j;
          } else if ( j == 0 ) {
            d = i;
          } else {
            d = 0.0;
          }
          // LOG(INFO) << "Initx: ga(" << i << "," << j << ")=" << d;
        });
        on_all_cores( [] {
            for( int i = 0; i < FLAGS_n; ++i ) {
              if( (Grappa::mycore() == 0) || 
                  (Grappa::mycore() == 1 && ga.dim2_percore == 1 ) ) {
                lefts[i].writeXF(i);
              } else {
                if( i == 0 ) {
                  lefts[i].writeXF( ga.local_chunk[0] - 1 ); // one to the left of our first element
                } else {
                  lefts[i].reset();
                }
              }
            }
          } );

      GlobalAddress<Timer> timer = symmetric_global_alloc<Timer>();

      LOG(INFO) << "Running " << FLAGS_iterations << " iterations....";

      for( int iter = 0; iter <= FLAGS_iterations; ++iter ) {

        on_all_cores( [timer, iter] {
            for( int i = 1; i < FLAGS_n; ++i ) {
              if( ! ((Grappa::mycore() == 0) || 
                     (Grappa::mycore() == 1 && ga.dim2_percore == 1 )) ) {
                lefts[i].reset();
              }
            }
            // ONLY DO THIS AFTER BARRIER
	    if (iter==1) timer->start = Grappa::walltime();

          } );
        
        // execute kernel
        VLOG(2) << "Starting iteration " << iter;


          finish( [timer] {
              on_all_cores( [timer] {
                  local = ga.local_chunk;
                  dim1_size = ga.dim1_size;
                  dim2_size = ga.dim2_size;
                  dim2_percore = ga.dim2_percore;
                  int first_j = Grappa::mycore() * dim2_percore;

                  for( int i = 1; i < dim1_size; ++i ) {

                    // prepare to iterate over this segment      
                    double left = readFF( &lefts[i] );
                    double diag = readFF( &lefts[i-1] );
                    double up = 0.0;
                    double current = 0.0;

                    for( int j = 0; j < dim2_percore; ++j ) {
                      int actual_j = j + first_j;
                      if( actual_j > 0 ) {
                        // compute this cell's value
                        up = local[ (i-1)*dim2_percore + j ];
                        current = up + left - diag;

                        // update for next iteration
                        diag = up;
                        left = current;

                        // write value
                        local[ (i)*dim2_percore + j ] = current;
                      }
                    }

                    // if we're at the end of a segment, write to corresponding full bit
                    if( Grappa::mycore()+1 < Grappa::cores() ) {
                      delegate::call<async>( Grappa::mycore()+1,
                                             [=] () {
                                               writeXF( &lefts[i], current );
                                             } );
                    }

                  }
                } );
            } );

        // copy top right corner value to bottom left corner to create dependency
        int last_m = FLAGS_m-1;
        int last_n = FLAGS_n-1;
        double val = delegate::read( &ga[ FLAGS_m-1 ][ FLAGS_n-1 ] );
        delegate::write( &ga[0][0], -1.0 * val );
        delegate::call( 0, [val] { lefts[0].writeXF( -1.0 * val ); } );
        if( ga.dim2_percore == 1 ) delegate::call( 1, [val] { lefts[0].writeXF( -1.0 * val ); } );
        on_all_cores ( [timer] {
          timer->total = Grappa::walltime() - timer->start;
	});
      }

      double iter_time;
      iter_time = reduce<double,collective_max<double>>( &timer->total );
      avgtime = iter_time/FLAGS_iterations;

      //      avgtime /= (double) std::max( FLAGS_iterations-1, static_cast<google::uint64>(1) );
      LOG(INFO) << "Rate (MFlops/s): " << 1.0E-06 * 2 * ((double)(FLAGS_m-1)*(FLAGS_n-1)) / avgtime;
      
      // verify result
      double expected_corner_val = (double) (FLAGS_iterations+1) * ( FLAGS_m + FLAGS_n - 2 );
      double actual_corner_val = delegate::read( &ga[ FLAGS_m-1 ][ FLAGS_n-1 ] );
      CHECK_DOUBLE_EQ( actual_corner_val, expected_corner_val );

      on_all_cores( [] {
          if( lefts ) delete [] lefts;
        } );
      leftsa.deallocate( );
      ga.deallocate( );
      
      LOG(INFO) << "Done.";
    });
  finalize();
  return 0;
}

