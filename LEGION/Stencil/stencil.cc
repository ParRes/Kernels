/*
Copyright (c) 2015, Intel Corporation

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

NAME:    stencil

PURPOSE: This program tests the efficiency with which a space-invariant,
         linear, symmetric filter (stencil) can be applied to a square
         grid or image.

USAGE:   The program takes as input the linear dimension of the grid, 
         and the number of iterations on the grid
 
         <progname> <# threads> <# iterations> <array dimension>
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.
 
         An optional parameter specifies the tile size used to divide the
         individual matrix blocks for improved cache and TLB performance.

         The output consists of diagnostics to make sure the
         transpose worked and timing statistics.

HISTORY: Written by Abdullah Kayi, May 2015.

*******************************************************************/
#include "../../include/par-res-kern_legion.h"

#include <sys/time.h>
#define  USEC_TO_SEC   1.0e-6    /* to convert microsecs to secs */

#ifdef DOUBLE
  #define DTYPE   double
  #define EPSILON 1.e-8
  #define COEFX   1.0
  #define COEFY   1.0
  #define FSTR    "%lf"
#else
  #define DTYPE   float
  #define EPSILON 0.0001f
  #define COEFX   1.0f
  #define COEFY   1.0f
  #define FSTR    "%f"
#endif

enum TaskIDs {
  TASKID_TOPLEVEL = 1,
  TASKID_SPMD,
  TASKID_WEIGHT_INITIALIZE,
  TASKID_INITIALIZE,
  TASKID_STENCIL,
  TASKID_DUMMY,
  TASKID_CHECK,
};

enum {
  FID_VAL,
  FID_DERIV,
  FID_GHOST,
  FID_WEIGHT,
};

enum {
  GHOST_LEFT,
  GHOST_NORTH,
  GHOST_RIGHT,
  GHOST_SOUTH,
};

struct SPMDArgs {
public:
  PhaseBarrier notify_ready[4];
  PhaseBarrier notify_empty[4];
  PhaseBarrier wait_ready[4];
  PhaseBarrier wait_empty[4];
  int region_idx[4];
  unsigned num_regions;
  int iterations;
  unsigned x_divs;
  unsigned y_divs;
  unsigned MYTHREAD;
  int x;
  int y;
  int n;
};

double wtime() {
  double time_seconds;

  struct timeval  time_data; /* seconds since 0 GMT */

  gettimeofday(&time_data,NULL);

  time_seconds  = (double) time_data.tv_sec;
  time_seconds += (double) time_data.tv_usec * USEC_TO_SEC;

  return time_seconds;
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int n;
  int x_divs = INT_MAX, y_divs = INT_MAX, threads;
  int iterations;

  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/
  const InputArgs &inputs = HighLevelRuntime::get_input_args();

  if (inputs.argc < 4){
    printf("Usage: %s <# threads> <# iterations> <array dimension>\n",
           *inputs.argv);
    exit(EXIT_FAILURE);
  }

  threads = atoi((inputs.argv[1]));
  if (threads <= 0){
      printf("ERROR: Number of THREADS must be > 0 : %d \n", threads);
      exit(EXIT_FAILURE);
  }

  iterations = atoi((inputs.argv[2]));
  if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n", iterations);
      exit(EXIT_FAILURE);
  }

  n = atoi(inputs.argv[3]);
  if (n <= 0){
      printf("ERROR: Matrix Order must be greater than 0 : %d \n", n);
      exit(EXIT_FAILURE);
  }

  printf("Parallel Research Kernels Version %s\n", PRKVERSION);
  printf("Legion Stencil Execution on 2D grid\n");
  printf("Grid size              = %d\n", n);
  printf("Number of threads      = %d\n", threads);
  printf("Radius of stencil      = %d\n", RADIUS);
#ifdef DOUBLE
  printf("Data type              = double precision\n");
#else
  printf("Data type              = single precision\n");
#endif
  printf("Number of iterations   = %d\n", iterations);

  /*********************************************************************
  ** Create the master index space
  *********************************************************************/

  int lower_bounds[] = {0, 0};
  int upper_bounds[] = {n - 1, n - 1};
  Point<2> lower_bounds_point(lower_bounds);
  Point<2> upper_bounds_point(upper_bounds);
  Rect<2> elem_rect(lower_bounds_point, upper_bounds_point);
  Domain domain = Domain::from_rect<2>(elem_rect);
  IndexSpace is = runtime->create_index_space(ctx, domain);

  /*********************************************************************
  ** Create ghost field space
  *********************************************************************/

  FieldSpace ghost_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, ghost_fs);
    allocator.allocate_field(sizeof(double),FID_GHOST);
  }

  /* x_divs=0 refers to automated calculation of division on each coordinates like MPI code */
  for (x_divs=(int) (sqrt(threads+1)); x_divs>0; x_divs--) {
      if (!(threads%x_divs)) {
        y_divs = threads/x_divs;
        break;
      }
    }

  if(threads % x_divs != 0){
    printf("THREADS %% x_divs != 0 (%d)\n", x_divs);
    exit(EXIT_FAILURE);
  }

  if(RADIUS < 1){
    printf("Stencil radius %d should be positive\n", RADIUS);
    exit(EXIT_FAILURE);
  }

  if(2*RADIUS +1 > n){
    printf("Stencil radius %d exceeds grid size %d\n", RADIUS, n);
    exit(EXIT_FAILURE);
  }

  if(n%threads != 0){
    printf("n%%THREADS should be zero\n");
    exit(EXIT_FAILURE);
  }

  printf("Tiles in x/y-direction = %d/%d\n", x_divs, y_divs);

  /*********************************************************************
  ** Create partitions for blocks
  *********************************************************************/
  int color_lower_bounds[] = {0, 0};
  int color_upper_bounds[] = {x_divs - 1, y_divs - 1};

  Point<2> color_lower_bounds_point(color_lower_bounds);
  Point<2> color_upper_bounds_point(color_upper_bounds);
  Rect<2> color_bounds(color_lower_bounds_point, color_upper_bounds_point);

  IndexPartition disjoint_ip;
  {
    int block_sizes[] = {n / x_divs, n / y_divs};
    Point<2> block_sizes_point(block_sizes);
    Blockify<2> coloring(block_sizes_point);
    disjoint_ip = runtime->create_index_partition(ctx, is, coloring);
  }

  // Now iterate over each of the sub-regions and make the ghost partitions
  Rect<1> ghost_bounds(Point<1>((int)GHOST_LEFT),Point<1>((int)GHOST_SOUTH));
  Domain ghost_domain = Domain::from_rect<1>(ghost_bounds);

  std::vector<LogicalRegion> ghost_left;
  std::vector<LogicalRegion> ghost_north;
  std::vector<LogicalRegion> ghost_right;
  std::vector<LogicalRegion> ghost_south;
  FieldSpace main_fs;
  std::vector<LogicalRegion> main_lr;

  for (GenericPointInRectIterator<2> itr(color_bounds); itr; itr++)
  {
    // Get each of the subspaces and create a main logical region from each
    // each subspace is a 2D block that belongs to a thread
    IndexSpace subspace = runtime->get_index_subspace(ctx, disjoint_ip, itr.p);
    Domain dom = runtime->get_index_space_domain(ctx, subspace);
    Rect<2> rect = dom.get_rect<2>();

    main_fs = runtime->create_field_space(ctx);
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, main_fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    allocator.allocate_field(sizeof(double),FID_DERIV);
    main_lr.push_back(runtime->create_logical_region(ctx, subspace, main_fs));

    // Make four sub-regions: LEFT,NORTH,RIGHT,SOUTH
    DomainColoring ghost_coloring;

    int left_start_bounds[] = {rect.lo.x[0], rect.lo.x[1]};
    int left_end_bounds[] = {rect.lo.x[0]+(RADIUS-1), rect.hi.x[1]};
    Point<2> left_start_point(left_start_bounds);
    Point<2> left_end_point(left_end_bounds);
    Rect<2> left(left_start_point, left_end_point);
    Domain left_domain = Domain::from_rect<2>(left);
    ghost_coloring[GHOST_LEFT] = left_domain;

    int north_start_bounds[] = {rect.lo.x[0], rect.lo.x[1]};
    int north_end_bounds[] = {rect.hi.x[0], rect.lo.x[1]+(RADIUS-1)};
    Point<2> north_start_point(north_start_bounds);
    Point<2> north_end_point(north_end_bounds);
    Rect<2> north(north_start_point, north_end_point);
    Domain north_domain = Domain::from_rect<2>(north);
    ghost_coloring[GHOST_NORTH] = north_domain;

    int right_start_bounds[] = {rect.hi.x[0]-(RADIUS-1), rect.lo.x[1]};
    int right_end_bounds[] = {rect.hi.x[0], rect.hi.x[1]};
    Point<2> right_start_point(right_start_bounds);
    Point<2> right_end_point(right_end_bounds);
    Rect<2> right(right_start_point, right_end_point);
    Domain right_domain = Domain::from_rect<2>(right);
    ghost_coloring[GHOST_RIGHT] = right_domain;

    int south_start_bounds[] = {rect.lo.x[0], rect.hi.x[1]-(RADIUS-1)};
    int south_end_bounds[] = {rect.hi.x[0], rect.hi.x[1]};
    Point<2> south_start_point(south_start_bounds);
    Point<2> south_end_point(south_end_bounds);
    Rect<2> south(south_start_point, south_end_point);
    Domain south_domain = Domain::from_rect<2>(south);
    ghost_coloring[GHOST_SOUTH] = south_domain;

    IndexPartition ghost_ip =
      runtime->create_index_partition(ctx, subspace, ghost_domain,
                                      ghost_coloring, true/*disjoint*/);
    // Make explicit logical regions for each of the ghost spaces
    for (int idx = GHOST_LEFT; idx <= GHOST_SOUTH; idx++)
    {
      IndexSpace ghost_space = runtime->get_index_subspace(ctx, ghost_ip, idx);
      LogicalRegion ghost_lr =
        runtime->create_logical_region(ctx, ghost_space, ghost_fs);
      if (idx == GHOST_LEFT)
        ghost_left.push_back(ghost_lr);
      else if (idx == GHOST_NORTH)
        ghost_north.push_back(ghost_lr);
      else if (idx == GHOST_RIGHT)
        ghost_right.push_back(ghost_lr);
      else if (idx == GHOST_SOUTH)
        ghost_south.push_back(ghost_lr);
    }
  }

  // Create all of the phase barriers for this computation
  std::vector<PhaseBarrier> left_ready_barriers;
  std::vector<PhaseBarrier> left_empty_barriers;
  std::vector<PhaseBarrier> right_ready_barriers;
  std::vector<PhaseBarrier> right_empty_barriers;
  std::vector<PhaseBarrier> north_ready_barriers;
  std::vector<PhaseBarrier> north_empty_barriers;
  std::vector<PhaseBarrier> south_ready_barriers;
  std::vector<PhaseBarrier> south_empty_barriers;
  for (int color = 0; color < threads; color++)
  {
    left_ready_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    left_empty_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    right_ready_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    right_empty_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    north_ready_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    north_empty_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    south_ready_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    south_empty_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
  }

  // In order to guarantee that all of our spmd_tasks execute in parallel
  // we have to use a must epoch launcher.  This instructs the runtime
  // to check that all of the operations in the must epoch are capable of
  // executing in parallel making it possible for them to synchronize using
  // named barriers with potential deadlock.  If for some reason they
  // cannot run in parallel, the runtime will report an error and indicate
  // the cause of it.

  {
    MustEpochLauncher must_epoch_launcher;
    // Need a separate array for storing these until we call the runtime
    std::vector<SPMDArgs> args(threads);
    // For each of our parallel tasks launch off a task with the ghost regions
    // for its neighbors as well as our ghost regions and the  necessary phase
    // barriers.

    for (int mygridposy = 0; mygridposy < y_divs; ++mygridposy)
      for (int mygridposx = 0; mygridposx < x_divs; ++mygridposx)
      {
        int color = mygridposy * x_divs + mygridposx;

        args[color].notify_ready[GHOST_LEFT] = left_ready_barriers[color];
        args[color].notify_ready[GHOST_RIGHT] = right_ready_barriers[color];
        args[color].notify_ready[GHOST_NORTH] = north_ready_barriers[color];
        args[color].notify_ready[GHOST_SOUTH] = south_ready_barriers[color];

        args[color].wait_empty[GHOST_LEFT] = left_empty_barriers[color];
        args[color].wait_empty[GHOST_RIGHT] = right_empty_barriers[color];
        args[color].wait_empty[GHOST_NORTH] = north_empty_barriers[color];
        args[color].wait_empty[GHOST_SOUTH] = south_empty_barriers[color];

        {
          int neighbor_x = mygridposx - 1;
          if (neighbor_x >= 0){
            int neighbor_color = mygridposy * x_divs + neighbor_x;
            args[color].wait_ready[GHOST_LEFT] = right_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_LEFT] = right_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_LEFT] = 0;
          }
          else
            args[color].region_idx[GHOST_LEFT] = -1;
        }
        {
          int neighbor_x = mygridposx + 1;
          if (neighbor_x < x_divs){
            int neighbor_color = mygridposy * x_divs + neighbor_x;
            args[color].wait_ready[GHOST_RIGHT] = left_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_RIGHT] = left_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_RIGHT] = 0;
          }
          else
            args[color].region_idx[GHOST_RIGHT] = -1;
        }
        {
          int neighbor_y = mygridposy - 1;
          if (neighbor_y >= 0){
            int neighbor_color = neighbor_y * x_divs + mygridposx;
            args[color].wait_ready[GHOST_NORTH] = south_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_NORTH] = south_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_NORTH] = 0;
          }
          else
            args[color].region_idx[GHOST_NORTH] = -1;
        }
        {
          int neighbor_y = mygridposy + 1;
          if (neighbor_y < y_divs){
            int neighbor_color = neighbor_y * x_divs + mygridposx;
            args[color].wait_ready[GHOST_SOUTH] = north_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_SOUTH] = north_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_SOUTH] = 0;
          }
          else
            args[color].region_idx[GHOST_SOUTH] = -1;
        }

        args[color].iterations = iterations;
        args[color].x_divs = x_divs;
        args[color].y_divs = y_divs;
        args[color].x = mygridposx;
        args[color].y = mygridposy;
        args[color].n = n;
        args[color].MYTHREAD = color;

        TaskLauncher spmd_launcher(TASKID_SPMD,
            TaskArgument(&args[color], sizeof(SPMDArgs)));

        // Main Block Region
        spmd_launcher.add_region_requirement(
            RegionRequirement(main_lr[color], READ_WRITE,
              SIMULTANEOUS, main_lr[color]));
        spmd_launcher.region_requirements[0].flags |= NO_ACCESS_FLAG;

        // Our Left
        if (mygridposx != 0)
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_left[color], READ_WRITE,
                SIMULTANEOUS, ghost_left[color]));
        // Our North
        if (mygridposy != 0)
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_north[color], READ_WRITE,
                SIMULTANEOUS, ghost_north[color]));
        // Our Right
        if (mygridposx != (x_divs-1))
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_right[color], READ_WRITE,
                SIMULTANEOUS, ghost_right[color]));
        // Our South
        if (mygridposy != y_divs-1)
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_south[color], READ_WRITE,
                SIMULTANEOUS, ghost_south[color]));

        // Left Ghost
        {
          int neighbor_x = mygridposx - 1;
          if (neighbor_x >= 0) {
            int neighbor_color = mygridposy * x_divs + neighbor_x;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_right[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_right[neighbor_color]));
          }
        }

        // North Ghost
        {
          int neighbor_y = mygridposy - 1;
          if (neighbor_y >= 0) {
            int neighbor_color = neighbor_y * x_divs + mygridposx;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_south[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_south[neighbor_color]));
          }
        }

        // Right Ghost
        {
          int neighbor_x = mygridposx + 1;
          if (neighbor_x < x_divs) {
            int neighbor_color = mygridposy * x_divs + neighbor_x;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_left[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_left[neighbor_color]));
          }
        }

        // South Ghost
        {
          int neighbor_y = mygridposy + 1;
          if (neighbor_y < y_divs) {
            int neighbor_color = neighbor_y * x_divs + mygridposx;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_north[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_north[neighbor_color]));
          }
        }

        spmd_launcher.add_field(0, FID_DERIV);
        spmd_launcher.add_field(0, FID_VAL);
        int ghost_count = std::count (args[color].region_idx, args[color].region_idx+4, 0);
        args[color].num_regions = (ghost_count * 2) + 1 ;

        for (unsigned idx = 1; idx < args[color].num_regions; idx++){
          spmd_launcher.add_field(idx, FID_GHOST);
          spmd_launcher.region_requirements[0].flags |= NO_ACCESS_FLAG;
        }

        spmd_launcher.add_index_requirement(IndexSpaceRequirement(is,
              NO_MEMORY,
              is));

        DomainPoint point(color);
        must_epoch_launcher.add_single_task(point, spmd_launcher);
      }
    FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
    fm.wait_all_results();

    typedef std::pair<double, double> pairdd;
    double ts_start = DBL_MAX, ts_end = DBL_MIN;
    std::pair<double, double> times(ts_start, ts_end);

    for (int color = 0; color < threads; color++){
      std::pair<double, double> times(fm.get_result<pairdd>(color));
      ts_start = MIN(ts_start, times.first);
      ts_end = MAX(ts_end, times.second);
    }

    double max_time = ts_end - ts_start;
    double avg_time = max_time / iterations;

    int stencil_size = 4*RADIUS+1;
    double f_active_points = (double) (n-2*RADIUS)*(double) (n-2*RADIUS);
    double flops = (DTYPE) (2*stencil_size+1) * f_active_points;

    printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
                   1.0E-06 * flops/avg_time, avg_time);

  }
  // Clean up our mess when we are done
  for (unsigned idx = 0; idx < main_lr.size(); idx++)
    runtime->destroy_logical_region(ctx, main_lr[idx]);
  for (unsigned idx = 0; idx < ghost_left.size(); idx++)
    runtime->destroy_logical_region(ctx, ghost_left[idx]);
  for (unsigned idx = 0; idx < ghost_right.size(); idx++)
    runtime->destroy_logical_region(ctx, ghost_north[idx]);
  for (unsigned idx = 0; idx < ghost_right.size(); idx++)
    runtime->destroy_logical_region(ctx, ghost_right[idx]);
  for (unsigned idx = 0; idx < ghost_right.size(); idx++)
    runtime->destroy_logical_region(ctx, ghost_south[idx]);

  for (unsigned idx = 0; idx < left_ready_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, left_ready_barriers[idx]);
  for (unsigned idx = 0; idx < left_empty_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, left_empty_barriers[idx]);
  for (unsigned idx = 0; idx < north_ready_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, north_ready_barriers[idx]);
  for (unsigned idx = 0; idx < north_empty_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, north_empty_barriers[idx]);
  for (unsigned idx = 0; idx < right_ready_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, right_ready_barriers[idx]);
  for (unsigned idx = 0; idx < right_empty_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, right_empty_barriers[idx]);
  for (unsigned idx = 0; idx < south_ready_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, south_ready_barriers[idx]);
  for (unsigned idx = 0; idx < south_empty_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, south_empty_barriers[idx]);

  ghost_left.clear();
  ghost_north.clear();
  ghost_right.clear();
  ghost_south.clear();

  left_ready_barriers.clear();
  left_empty_barriers.clear();
  north_ready_barriers.clear();
  north_empty_barriers.clear();
  right_ready_barriers.clear();
  right_empty_barriers.clear();
  south_ready_barriers.clear();
  south_empty_barriers.clear();

  runtime->destroy_index_space(ctx, is);
  runtime->destroy_field_space(ctx, ghost_fs);
}

std::pair<double, double> spmd_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime)
{
  SPMDArgs *args = (SPMDArgs*)task->args;

  // local_lr that holds the entire block of data
  LogicalRegion local_lr = task->regions[0].region;

  std::vector<LogicalRegion> neighbor_lr;
  std::vector<LogicalRegion> ghost_lr;
  std::vector<PhysicalRegion> ghost_pr;
  std::vector<PhaseBarrier> notify_ready;
  std::vector<PhaseBarrier> notify_empty;
  std::vector<PhaseBarrier> wait_ready;
  std::vector<PhaseBarrier> wait_empty;

  assert((regions.size() - 1) % 2 == 0);
  unsigned num_neighbors = (regions.size() - 1) / 2;

  unsigned idx = 0;
  for (unsigned dir = GHOST_LEFT; dir <= GHOST_SOUTH; ++dir){
    if (args->region_idx[dir] != -1)
    {
      neighbor_lr.push_back(task->regions[1 + idx].region);
      ghost_lr.push_back(task->regions[1 + num_neighbors + idx].region);
      ghost_pr.push_back(regions[1 + num_neighbors + idx]);
      notify_ready.push_back(args->notify_ready[dir]);
      wait_ready.push_back(args->wait_ready[dir]);
      notify_empty.push_back(args->notify_empty[dir]);
      wait_empty.push_back(args->wait_empty[dir]);
      ++idx;
    }
  }

  //create logical region for WEIGHT matrix
  int weight_start[] = {-RADIUS, -RADIUS};
  int weight_end[] = {RADIUS, RADIUS};
  Point<2> weight_start_point(weight_start);
  Point<2> weight_end_point(weight_end);
  Rect<2> weight_rect(weight_start_point, weight_end_point);
  Domain domain_weight = Domain::from_rect<2>(weight_rect);
  IndexSpace is_weight = runtime->create_index_space(ctx, domain_weight);

  FieldSpace fs_weight = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_weight);
    allocator.allocate_field(sizeof(double),FID_WEIGHT);
  }
  LogicalRegion lr_weight = runtime->create_logical_region(ctx, is_weight, fs_weight);

  TaskLauncher weight_init_launcher(TASKID_WEIGHT_INITIALIZE,
                              TaskArgument(NULL, 0));
  weight_init_launcher.add_region_requirement(
      RegionRequirement(lr_weight, WRITE_DISCARD,
                        EXCLUSIVE, lr_weight));
  weight_init_launcher.add_field(0, FID_WEIGHT);
  runtime->execute_task(ctx, weight_init_launcher);

  // Launch a task to initialize our field with some data
  TaskLauncher init_launcher(TASKID_INITIALIZE,
                             TaskArgument(args, sizeof(SPMDArgs)));
  init_launcher.add_region_requirement(
      RegionRequirement(local_lr, WRITE_DISCARD,
                        EXCLUSIVE, local_lr));
  init_launcher.add_field(0, FID_VAL);
  init_launcher.add_region_requirement(
      RegionRequirement(local_lr, WRITE_DISCARD,
                        EXCLUSIVE, local_lr));
  init_launcher.add_field(1, FID_DERIV);
  runtime->execute_task(ctx, init_launcher);

  // Run a bunch of steps
  double ts_start = DBL_MAX, ts_end = DBL_MIN;
  for (int iter = 0; iter <= args->iterations; iter++)
  {
    if(iter == 1)
    {
      TaskLauncher dummy_launcher(TASKID_DUMMY, TaskArgument(NULL, 0));
      Future f = runtime->execute_task(ctx, dummy_launcher);
      f.get_void_result();
      ts_start = wtime();
    }

    // Issue explicit region-to-region copies
    for (unsigned idx = 0; idx < num_neighbors; idx++)
    {
      CopyLauncher copy_launcher;
      copy_launcher.add_copy_requirements(
          RegionRequirement(local_lr, READ_ONLY,
                            EXCLUSIVE, local_lr),
          RegionRequirement(neighbor_lr[idx], WRITE_DISCARD,
                            EXCLUSIVE, neighbor_lr[idx]));
      copy_launcher.add_src_field(0, FID_VAL);
      copy_launcher.add_dst_field(0, FID_GHOST);
      // It's not safe to issue the copy until we know
      // that the destination instance is empty. Only
      // need to do this after the first iteration.
      if (iter > 0)
      {
        wait_empty[idx] =
          runtime->advance_phase_barrier(ctx, wait_empty[idx]);
        copy_launcher.add_wait_barrier(wait_empty[idx]);
      }
      // When we are done with the copy, signal that the
      // destination instance is now ready
      copy_launcher.add_arrival_barrier(notify_ready[idx]);
      runtime->issue_copy_operation(ctx, copy_launcher);
      // Once we've issued our copy operation, advance both of
      // the barriers to the next generation.
      notify_ready[idx] =
        runtime->advance_phase_barrier(ctx, notify_ready[idx]);
    }

    // Acquire coherence on our ghost regions
    for (unsigned idx = 0; idx < num_neighbors; idx++)
    {
      AcquireLauncher acquire_launcher(ghost_lr[idx],
                                       ghost_lr[idx],
                                       ghost_pr[idx]);
      acquire_launcher.add_field(FID_GHOST);
      // The acquire operation needs to wait for the data to
      // be ready to consume, so wait on the ready barrier.
      wait_ready[idx] = runtime->advance_phase_barrier(ctx, wait_ready[idx]);
      acquire_launcher.add_wait_barrier(wait_ready[idx]);
      runtime->issue_acquire(ctx, acquire_launcher);
    }

    // Run the stencil computation
    TaskLauncher stencil_launcher(TASKID_STENCIL,
                                  TaskArgument(args, sizeof(SPMDArgs)));
    stencil_launcher.add_region_requirement(
        RegionRequirement(local_lr, READ_WRITE, EXCLUSIVE, local_lr));
    stencil_launcher.add_field(0, FID_DERIV);
    stencil_launcher.add_region_requirement(
        RegionRequirement(local_lr, READ_WRITE, EXCLUSIVE, local_lr));
    stencil_launcher.add_field(1, FID_VAL);

    // add region for WEIGHT matrix
    {
      RegionRequirement req(lr_weight, READ_ONLY, EXCLUSIVE, lr_weight);
      req.add_field(FID_WEIGHT);
      stencil_launcher.add_region_requirement(req);
    }

    for (unsigned idx = 0; idx < num_neighbors; idx++)
    {
      RegionRequirement ghost_req(ghost_lr[idx], READ_ONLY, EXCLUSIVE, ghost_lr[idx]);
      ghost_req.add_field(FID_GHOST);
      stencil_launcher.add_region_requirement(ghost_req);
    }
    runtime->execute_task(ctx, stencil_launcher);

    // Release coherence on ghost regions
    for (unsigned idx = 0; idx < num_neighbors; idx++)
    {
      ReleaseLauncher release_launcher(ghost_lr[idx],
                                       ghost_lr[idx],
                                       ghost_pr[idx]);
      release_launcher.add_field(FID_GHOST);
      // On all but the last iteration we need to signal that
      // we have now consumed the ghost instances and it is
      // safe to issue the next copy.
      if (iter < (args->iterations))
        release_launcher.add_arrival_barrier(notify_empty[idx]);
      runtime->issue_release(ctx, release_launcher);
      if (iter < (args->iterations))
        notify_empty[idx] =
          runtime->advance_phase_barrier(ctx, notify_empty[idx]);
    }
  }
  ts_end = wtime();

  //runtime->issue_execution_fence(ctx);
  TaskLauncher dummy_launcher(TASKID_DUMMY, TaskArgument(NULL, 0));
  Future f = runtime->execute_task(ctx, dummy_launcher);
  f.get_void_result();

  TaskLauncher check_launcher(TASKID_CHECK, TaskArgument(args, sizeof(SPMDArgs)));
  check_launcher.add_region_requirement(
      RegionRequirement(local_lr, READ_ONLY,
        EXCLUSIVE, local_lr));
  check_launcher.add_field(0, FID_DERIV);
  runtime->execute_task(ctx, check_launcher);

  return std::pair<double, double>(ts_start, ts_end);
}

void dummy_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  //empty
}

void init_weight_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  RegionAccessor<AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(FID_WEIGHT).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    Point<2> p = pir.p;
    int xx = p.x[0];
    int yy = p.x[1];

    if(yy == 0){
      if(xx == 0)
        continue;
      else{
        DTYPE val = (1.0/(2.0*xx*RADIUS));
        acc.write(DomainPoint::from_point<2>(p), val);
      }
    }
    else if(xx == 0){
      if(yy == 0)
        continue;
      else{
        DTYPE val = (1.0/(2.0*yy*RADIUS));
        acc.write(DomainPoint::from_point<2>(p), val);
      }
    }
    else
      acc.write(DomainPoint::from_point<2>(pir.p), (DTYPE) 0.0);
  }

}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);

  RegionAccessor<AccessorType::Generic, double> in_acc =
    regions[0].get_field_accessor(FID_VAL).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> out_acc =
    regions[1].get_field_accessor(FID_DERIV).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
      double value = COEFY*pir.p.x[1] + COEFX*pir.p.x[0];
      in_acc.write(DomainPoint::from_point<2>(pir.p), value);
      out_acc.write(DomainPoint::from_point<2>(pir.p), 0.0);
      //printf("(%d,%d)=%f\n", pir.p.x[0], pir.p.x[1], value);
  }
}

void stencil_field_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  unsigned num_neighbors = regions.size() - 3;

  SPMDArgs *args = (SPMDArgs*)task->args;

  FieldID write_fid = *(task->regions[0].privilege_fields.begin());
  FieldID read_fid = *(task->regions[1].privilege_fields.begin());
  FieldID weight_fid = *(task->regions[2].privilege_fields.begin());
  FieldID ghost_fid;

  if(num_neighbors > 0) {
    ghost_fid = *(task->regions[3].privilege_fields.begin());
  } else {
    ghost_fid = INT_MAX; /* silence uninitialized warning */
  }

  RegionAccessor<AccessorType::Generic, double> write_acc =
    regions[0].get_field_accessor(write_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> read_acc =
    regions[1].get_field_accessor(read_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> weight_acc =
    regions[2].get_field_accessor(weight_fid).typeify<double>();

  Domain main_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain weight_dom = runtime->get_index_space_domain(ctx,
      task->regions[2].region.get_index_space());
  Rect<2> main_rect = main_dom.get_rect<2>();
  Rect<2> weight_rect = weight_dom.get_rect<2>();

  std::vector<PhysicalRegion> stencil_ghosts_pr;
  std::vector<LogicalRegion> stencil_ghosts_lr;
  for (unsigned idx = 0; idx < num_neighbors; idx++)
  {
    stencil_ghosts_pr.push_back(regions[3 + idx]);
    stencil_ghosts_lr.push_back(task->regions[3 + idx].region);
  }

  double* write_ptr = 0;
  double* read_ptr = 0;
  double* weight_ptr = 0;
  double* left_ptr = 0;
  double* north_ptr = 0;
  double* right_ptr = 0;
  double* south_ptr = 0;

  {
      Rect<2> subrect_a; ByteOffset offsets_a[1];
      write_ptr = write_acc.raw_rect_ptr<2>(main_rect, subrect_a, offsets_a);
      read_ptr = read_acc.raw_rect_ptr<2>(main_rect, subrect_a, offsets_a);
      assert(main_rect == subrect_a);
  }
  {
      Rect<2> subrect_w; ByteOffset offsets_w[1];
      weight_ptr = weight_acc.raw_rect_ptr<2>(weight_rect, subrect_w, offsets_w);
      assert(weight_rect == subrect_w);
  }
  Domain left_dom;
  Domain north_dom;
  Domain right_dom;
  Domain south_dom;
  Rect<2> left_rect;
  Rect<2> north_rect;
  Rect<2> right_rect;
  Rect<2> south_rect;
  RegionAccessor<AccessorType::Generic, double> left_ghost_acc;
  RegionAccessor<AccessorType::Generic, double> north_ghost_acc;
  RegionAccessor<AccessorType::Generic, double> right_ghost_acc;
  RegionAccessor<AccessorType::Generic, double> south_ghost_acc;

  /* UNUSED
   * unsigned MYTHREAD = args->MYTHREAD;
   */

  unsigned idx = 0;
  if (args->region_idx[GHOST_LEFT] != -1){
      left_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<double>();
      left_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      left_rect = left_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      left_ptr = left_ghost_acc.raw_rect_ptr<2>(left_rect, subrect_g, offsets_g);
      assert(left_rect == subrect_g);
      ++idx;
  }
  if (args->region_idx[GHOST_NORTH] != -1){
      north_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<double>();
      north_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      north_rect = north_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      north_ptr = north_ghost_acc.raw_rect_ptr<2>(north_rect, subrect_g, offsets_g);
      assert(north_rect == subrect_g);
      ++idx;
  }
  if (args->region_idx[GHOST_RIGHT] != -1){
      right_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<double>();
      right_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      right_rect = right_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      right_ptr = right_ghost_acc.raw_rect_ptr<2>(right_rect, subrect_g, offsets_g);
      assert(right_rect == subrect_g);
      ++idx;
  }
  if (args->region_idx[GHOST_SOUTH] != -1){
      south_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<double>();
      south_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      south_rect = south_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      south_ptr = south_ghost_acc.raw_rect_ptr<2>(south_rect, subrect_g, offsets_g);
      assert(south_rect == subrect_g);
  }

  /* UNUSED
   * int lower_y = main_rect.lo.x[1];
   * int lower_x = main_rect.lo.x[0];
   * int upper_y = main_rect.hi.x[1];
   * int upper_x = main_rect.hi.x[0];
   */

  int x_divs = args->x_divs;
  int y_divs = args->y_divs;
  int blockx = args->n / x_divs;
  int blocky = args->n / y_divs;
  int mygridposx = args->x;
  int mygridposy = args->y;
  int myoffsetx = mygridposx * blockx;
  int myoffsety = mygridposy * blocky;

  int startx = 0;
  int starty = 0;
  int endx = blockx-1;
  int endy = blocky-1;

  if(mygridposx == 0)        startx += RADIUS;
  if(mygridposy == 0)        starty += RADIUS;
  if(mygridposx == x_divs-1) endx   -= RADIUS;
  if(mygridposy == y_divs-1) endy   -= RADIUS;

  int start_idx = (starty*blockx) + startx;
  int end_idx = (endy*blockx) + endx;
  /* UNUSED
   * int grid_size = (endx-startx+1) * (endy-starty+1);
   */

  int sizew = 2*RADIUS+1;

#define WEIGHT_X(i) weight_ptr[i+(sizew*RADIUS+RADIUS)]
#define WEIGHT_Y(i) weight_ptr[i+(sizew-1)*(i+RADIUS+1)]
  for(int i = start_idx; i <= end_idx; i++)
  {
      int x = i % blockx;
      int y = i / blockx;
      int real_x = myoffsetx + x;
      int real_y = myoffsety + y;

      int real_coords[2] = {real_x,real_y};
      Point<2> real_pos(real_coords);
      if(real_pos.x[0] < RADIUS || real_pos.x[1] < RADIUS)
          continue;
      if(real_pos.x[0] >= (args->n - RADIUS) || real_pos.x[1] >= (args->n - RADIUS))
          continue;

      double val = 0.0;

      for(int xx = -RADIUS; xx <= RADIUS; xx++)
      {
          int xcoords[2] = {xx, 0};
          Point<2> r(xcoords);

          Point<2> p2 = real_pos + r;

          if (main_rect.contains(p2)){
              val += WEIGHT_X(xx) * read_ptr[i+xx];
          }
          else if (args->region_idx[GHOST_LEFT] != -1 && left_rect.contains(p2)){
              int left_idx = (x + RADIUS + xx) + y * RADIUS;
              val += WEIGHT_X(xx) * left_ptr[left_idx];
          }
          else if (args->region_idx[GHOST_RIGHT] != -1 && right_rect.contains(p2)){
              int right_idx = (x + xx - blockx) + y * RADIUS;
              val += WEIGHT_X(xx) * right_ptr[right_idx];
          }
          else{
              assert(false);
          }
      }

      for(int yy = -RADIUS; yy <= RADIUS; yy++)
      {
          int ycoords[2] = {0, yy};
          Point<2> r(ycoords);

          Point<2> p2 = real_pos + r;

          if (main_rect.contains(p2)){
              val += WEIGHT_Y(yy) * read_ptr[i+(yy*blockx)];
          }
          else if (args->region_idx[GHOST_NORTH] != -1 && north_rect.contains(p2)){
              int north_idx = x + (RADIUS + y + yy) * blockx;
              val += WEIGHT_Y(yy) * north_ptr[north_idx];
          }
          else if (args->region_idx[GHOST_SOUTH] != -1 && south_rect.contains(p2)){
              int south_idx = x + (y + yy - blocky) * blockx;
              val += WEIGHT_Y(yy) * south_ptr[south_idx];
          }
          else{
              assert(false);
          }
      }

      double current_val = write_ptr[i];
      double output_val = current_val + val;
      write_ptr[i] = output_val;
  }
 
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  SPMDArgs *args = (SPMDArgs*)task->args;

  RegionAccessor<AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(FID_DERIV).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  double abserr = 0.0;
  double norm = 0.0;
  double value = 0.0;
  double epsilon=1.e-8; /* error tolerance */

  double* acc_ptr = 0;
  {
    Rect<2> subrect_a; ByteOffset offsets_a[1];
    acc_ptr =acc.raw_rect_ptr<2>(rect, subrect_a, offsets_a);
    assert(rect == subrect_a);
  }

  int x_divs = args->x_divs;
  int y_divs = args->y_divs;
  int blockx = args->n / x_divs;
  int blocky = args->n / y_divs;
  int mygridposx = args->x;
  int mygridposy = args->y;
  int myoffsetx = mygridposx * blockx;
  int myoffsety = mygridposy * blocky;

  int startx = 0;
  int starty = 0;
  int endx = blockx-1;
  int endy = blocky-1;

  if(mygridposx == 0)        startx += RADIUS;
  if(mygridposy == 0)        starty += RADIUS;
  if(mygridposx == x_divs-1) endx   -= RADIUS;
  if(mygridposy == y_divs-1) endy   -= RADIUS;

  int start_idx = (starty*blocky) + startx;
  int end_idx = (endy*blocky) + endx;

  for(int i = start_idx; i <= end_idx; i++)
  {
    int x = i % blockx;
    int y = i / blockx;
    int real_x = myoffsetx + x;
    int real_y = myoffsety + y;

    int real_coords[2] = {real_x,real_y};
    Point<2> real_pos(real_coords);
    if(real_pos.x[0] < RADIUS || real_pos.x[1] < RADIUS)
      continue;
    if(real_pos.x[0] >= (args->n - RADIUS) || real_pos.x[1] >= (args->n - RADIUS))
      continue;

    norm = (DTYPE) (args->iterations + 1) * (COEFX + COEFY);
    value = acc_ptr[i];
    abserr += ABS(value - norm);
  }

  if (abserr < epsilon) {
    if (args->MYTHREAD == 0) printf("Solution validates\n");
#ifdef VERBOSE
    if (args->MYTHREAD == 0) printf("Squared errors: %f \n", abserr);
#endif
  }
  else {
    if (args->MYTHREAD == 0) printf("ERROR: Squared error %lf exceeds threshold %e\n",
        abserr, epsilon);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TASKID_TOPLEVEL);
  HighLevelRuntime::register_legion_task<top_level_task>(TASKID_TOPLEVEL,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "top_level");
  HighLevelRuntime::register_legion_task<std::pair<double,double>, spmd_task>(TASKID_SPMD,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "spmd");
  HighLevelRuntime::register_legion_task<init_weight_task>(TASKID_WEIGHT_INITIALIZE,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "init_weight");
  HighLevelRuntime::register_legion_task<init_field_task>(TASKID_INITIALIZE,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "init");
  HighLevelRuntime::register_legion_task<stencil_field_task>(TASKID_STENCIL,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "stencil");
  HighLevelRuntime::register_legion_task<dummy_task>(TASKID_DUMMY,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "check");
  HighLevelRuntime::register_legion_task<check_task>(TASKID_CHECK,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "check");

  return HighLevelRuntime::start(argc, argv);
}
