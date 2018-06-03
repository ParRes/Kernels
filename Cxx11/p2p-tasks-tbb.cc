///
/// Copyright (c) 2013, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an m*n grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <m> <n>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///            C99-ification by Jeff Hammond, February 2016.
///            C++11-ification by Jeff Hammond, May 2017.
///            TBB implementation by Pablo Reble, April 2018.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_tbb.h"
#include "p2p-kernel.h"

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/TBB Flow Graph pipeline execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  using namespace tbb::flow;
  //graph g;

  int iterations;
  int m, n;
  int mc, nc;
  try {
      if (argc < 4){
        throw " <# iterations> <first array dimension> <second array dimension> [<first chunk dimension> <second chunk dimension>]";
      }

      // number of times to run the pipeline algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // grid dimensions
      m = std::atoi(argv[2]);
      n = std::atoi(argv[3]);
      if (m < 1 || n < 1) {
        throw "ERROR: grid dimensions must be positive";
      } else if ( static_cast<size_t>(m)*static_cast<size_t>(n) > INT_MAX) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

      // grid chunk dimensions
      mc = (argc > 4) ? std::atoi(argv[4]) : m;
      nc = (argc > 5) ? std::atoi(argv[5]) : n;
      if (mc < 1 || mc > m || nc < 1 || nc > n) {
        std::cout << "WARNING: grid chunk dimensions invalid: " << mc <<  nc << " (ignoring)" << std::endl;
        mc = m;
        nc = n;
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  const char* envvar = std::getenv("TBB_NUM_THREADS");
  int num_threads = (envvar!=NULL) ? std::atoi(envvar) : tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init init(num_threads);

  std::cout << "Number of threads    = " << num_threads << std::endl;
  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << m << ", " << n << std::endl;
  std::cout << "Grid chunk sizes     = " << mc << ", " << nc << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Create Grid and allocate space
  //////////////////////////////////////////////////////////////////////
  // calculate number of tiles in n and m direction to create grid.
  int num_blocks_n = (n / nc);
  if(n%nc != 0) num_blocks_n++;
  int num_blocks_m = (m / mc);
  if(m%mc != 0) num_blocks_m++;

  auto pipeline_time = 0.0; // silence compiler warning

  double * grid = new double[m*n];

  typedef tbb::flow::continue_node< tbb::flow::continue_msg > block_node_t;

  graph g;
  block_node_t *nodes[ num_blocks_n * num_blocks_m ];
  // To enable tracing support for Flow Graph Analyzer
  // set following MACRO and link against TBB preview library (-ltbb_preview)
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
  char buffer[1024];
  g.set_name("Pipeline");
#endif

  bool first_iter=true;
  block_node_t b(g, [&](const tbb::flow::continue_msg &){
    grid[0*n+0] = -grid[(m-1)*n+(n-1)];
    if(first_iter) pipeline_time = prk::wtime();
      first_iter = false;
  });
  for (int i=0; i<num_blocks_m; i+=1) {
    for (int j=0; j<num_blocks_n; j+=1) {
        block_node_t *tmp = new block_node_t(g, [=](const tbb::flow::continue_msg &){
            sweep_tile((i*mc)+1, std::min(m,(i*mc)+mc+1), (j*nc)+1, std::min(n,(j*nc)+nc+1), n, grid);
        });
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
        sprintf(buffer, "block [ %d, %d ]", i, j );
        tmp->set_name( buffer );
#endif
        nodes[i*num_blocks_n + j] = tmp;
        if (i>0)
          make_edge(*nodes[(i-1)*num_blocks_n + j ], *tmp );
        if (j>0)
          make_edge(*nodes[ i   *num_blocks_n + j-1], *tmp );
        // Transitive dependencies from OpenMP task version:
        //make_edge( *tmp, b );
        //if (i>0 && j>0)
        //  make_edge(*nodes[(i-1)*num_blocks_n + j-1], *tmp );
    }
  }
  auto start = true;
  source_node<continue_msg> s(g, [&](continue_msg &v) -> bool {
    if(start) { 
      v = continue_msg();
      start = false;
      return true;
    }
    return false;
  }, false);
  
  limiter_node<continue_msg> l(g, iterations+1, 1);

  make_edge( s, l );
  make_edge( l, *nodes[0] );
  make_edge( *nodes[(num_blocks_n * num_blocks_m) - 1], b);
  make_edge( b, l );

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
  s.set_name("Source");
  b.set_name("Iteration Barrier");
  l.set_name("Limiter");
#endif

  //////////////////////////////////////////////////////////////////////
  // Perform the computation
  //////////////////////////////////////////////////////////////////////

  {

    tbb::blocked_range2d<int> range(0, m, mc, 0, n, nc);
    tbb::parallel_for( range, [&](decltype(range)& r) {
      for (auto i=r.rows().begin(); i!=r.rows().end(); ++i ) {
        for (auto j=r.cols().begin(); j!=r.cols().end(); ++j ) {
          grid[i*n+j] = 0.0;
        }
      }
    }, tbb_partitioner);
    for (auto j=0; j<n; j++) {
      grid[0*n+j] = static_cast<double>(j);
    }
    for (auto i=0; i<m; i++) {
      grid[i*n+0] = static_cast<double>(i);
    }

    s.activate();
    g.wait_for_all();
    
    pipeline_time = prk::wtime() - pipeline_time;

  }

  //////////////////////////////////////////////////////////////////////
  // Cleanup Flow Graph
  //////////////////////////////////////////////////////////////////////

  for (int i=0; i<num_blocks_m; i+=1) {
    for (int j=0; j<num_blocks_n; j+=1) {
      delete nodes[i*num_blocks_n + j];
    }
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  auto corner_val = ((iterations+1.)*(n+m-2.));
  if ( (std::fabs(grid[(m-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << grid[(m-1)*n+(n-1)]
              << " does not match verification value " << corner_val << std::endl;
    return 1;
  }

#ifdef VERBOSE
  std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
  std::cout << "Solution validates" << std::endl;
#endif
  auto avgtime = pipeline_time/iterations;
  std::cout << "Rate (MFlops/s): "
            << 2.0e-6 * ( (m-1.)*(n-1.) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;

  return 0;
}
