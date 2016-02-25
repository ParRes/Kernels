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

#define DOUBLE

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

class StencilMapper : public DefaultMapper
{
  public:
    StencilMapper(Machine machine, HighLevelRuntime *rt, Processor local);
    virtual bool map_must_epoch(const std::vector<Task*> &tasks,
        const std::vector<MappingConstraint> &constraints,
        MappingTagID tag);
  private:
    std::vector<Processor> all_procs_list;
    std::map<Processor, Memory> all_sysmems;
};

StencilMapper::StencilMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  machine_interface.filter_processors(machine, Processor::LOC_PROC, all_procs);

  for (std::set<Processor>::iterator itr = all_procs.begin();
      itr != all_procs.end(); ++itr)
  {
    Memory sysmem = machine_interface.find_memory_kind(*itr, Memory::SYSTEM_MEM);
    assert(sysmem.exists());
    all_sysmems[*itr] = sysmem;
  }
  all_procs_list.assign(all_procs.begin(), all_procs.end());
}

bool StencilMapper::map_must_epoch(const std::vector<Task*> &tasks,
    const std::vector<MappingConstraint> &constraints,
    MappingTagID tag)
{
  for (unsigned i = 0; i < tasks.size(); ++i)
  {
    Task* task = tasks[i];
    task->target_proc = all_procs_list[task->index_point.get_index()];
    map_task(task);
  }

  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    if (c.t1->target_proc.address_space() ==
        c.t2->target_proc.address_space()) continue;
    if (c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
    {
      assert((c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG) == 0);
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(all_sysmems[c.t1->target_proc]);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(all_sysmems[c.t1->target_proc]);
    }
    else if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG)
    {
      assert((c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG) == 0);
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(all_sysmems[c.t2->target_proc]);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(all_sysmems[c.t2->target_proc]);
    }
    else
      assert(0);
  }

  return false;
}

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
  PhaseBarrier global_barrier;
  int region_idx[4];
  unsigned num_regions;
  int iterations;
  unsigned Num_procsx;
  unsigned Num_procsy;
  unsigned my_ID;
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
  int Num_procsx, Num_procsy, threads;
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
    allocator.allocate_field(sizeof(DTYPE),FID_GHOST);
  }

  /* compute "processor" grid; initialize Num_procsy to avoid compiler warnings */
  Num_procsy = 0;
  for (Num_procsx=(int) (sqrt(threads+1)); Num_procsx>0; Num_procsx--) {
      if (!(threads%Num_procsx)) {
        Num_procsy = threads/Num_procsx;
        break;
      }
    }

  if(threads % Num_procsx != 0){
    printf("THREADS %% Num_procsx != 0 (%d)\n", Num_procsx);
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

  printf("Tiles in x/y-direction = %d/%d\n", Num_procsx, Num_procsy);

  /*********************************************************************
  ** Create partitions for blocks
  *********************************************************************/
  int color_lower_bounds[] = {0, 0};
  int color_upper_bounds[] = {Num_procsx - 1, Num_procsy - 1};

  Point<2> color_lower_bounds_point(color_lower_bounds);
  Point<2> color_upper_bounds_point(color_upper_bounds);
  Rect<2> color_bounds(color_lower_bounds_point, color_upper_bounds_point);

  IndexPartition disjoint_ip;
  {
    int block_sizes[] = {n / Num_procsx, n / Num_procsy};
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
    allocator.allocate_field(sizeof(DTYPE),FID_VAL);
    allocator.allocate_field(sizeof(DTYPE),FID_DERIV);
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
  PhaseBarrier global_barrier = runtime->create_phase_barrier(ctx, threads);

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

    for (int my_IDy = 0; my_IDy < Num_procsy; ++my_IDy)
      for (int my_IDx = 0; my_IDx < Num_procsx; ++my_IDx)
      {
        int color = my_IDy * Num_procsx + my_IDx;

        args[color].notify_ready[GHOST_LEFT] = left_ready_barriers[color];
        args[color].notify_ready[GHOST_RIGHT] = right_ready_barriers[color];
        args[color].notify_ready[GHOST_NORTH] = north_ready_barriers[color];
        args[color].notify_ready[GHOST_SOUTH] = south_ready_barriers[color];

        args[color].wait_empty[GHOST_LEFT] = left_empty_barriers[color];
        args[color].wait_empty[GHOST_RIGHT] = right_empty_barriers[color];
        args[color].wait_empty[GHOST_NORTH] = north_empty_barriers[color];
        args[color].wait_empty[GHOST_SOUTH] = south_empty_barriers[color];

        {
          int neighbor_x = my_IDx - 1;
          if (neighbor_x >= 0){
            int neighbor_color = my_IDy * Num_procsx + neighbor_x;
            args[color].wait_ready[GHOST_LEFT] = right_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_LEFT] = right_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_LEFT] = 0;
          }
          else
            args[color].region_idx[GHOST_LEFT] = -1;
        }
        {
          int neighbor_x = my_IDx + 1;
          if (neighbor_x < Num_procsx){
            int neighbor_color = my_IDy * Num_procsx + neighbor_x;
            args[color].wait_ready[GHOST_RIGHT] = left_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_RIGHT] = left_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_RIGHT] = 0;
          }
          else
            args[color].region_idx[GHOST_RIGHT] = -1;
        }
        {
          int neighbor_y = my_IDy - 1;
          if (neighbor_y >= 0){
            int neighbor_color = neighbor_y * Num_procsx + my_IDx;
            args[color].wait_ready[GHOST_NORTH] = south_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_NORTH] = south_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_NORTH] = 0;
          }
          else
            args[color].region_idx[GHOST_NORTH] = -1;
        }
        {
          int neighbor_y = my_IDy + 1;
          if (neighbor_y < Num_procsy){
            int neighbor_color = neighbor_y * Num_procsx + my_IDx;
            args[color].wait_ready[GHOST_SOUTH] = north_ready_barriers[neighbor_color];
            args[color].notify_empty[GHOST_SOUTH] = north_empty_barriers[neighbor_color];
            args[color].region_idx[GHOST_SOUTH] = 0;
          }
          else
            args[color].region_idx[GHOST_SOUTH] = -1;
        }

        args[color].iterations = iterations;
        args[color].Num_procsx = Num_procsx;
        args[color].Num_procsy = Num_procsy;
        args[color].x = my_IDx;
        args[color].y = my_IDy;
        args[color].n = n;
        args[color].my_ID = color;
        args[color].global_barrier = global_barrier;

        TaskLauncher spmd_launcher(TASKID_SPMD,
            TaskArgument(&args[color], sizeof(SPMDArgs)));

        // Main Block Region
        spmd_launcher.add_region_requirement(
            RegionRequirement(main_lr[color], READ_WRITE,
              SIMULTANEOUS, main_lr[color]));
        spmd_launcher.region_requirements[0].flags |= NO_ACCESS_FLAG;

        // Our Left
        if (my_IDx != 0)
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_left[color], READ_WRITE,
                SIMULTANEOUS, ghost_left[color]));
        // Our North
        if (my_IDy != 0)
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_north[color], READ_WRITE,
                SIMULTANEOUS, ghost_north[color]));
        // Our Right
        if (my_IDx != (Num_procsx-1))
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_right[color], READ_WRITE,
                SIMULTANEOUS, ghost_right[color]));
        // Our South
        if (my_IDy != Num_procsy-1)
          spmd_launcher.add_region_requirement(
              RegionRequirement(ghost_south[color], READ_WRITE,
                SIMULTANEOUS, ghost_south[color]));

        // Left Ghost
        {
          int neighbor_x = my_IDx - 1;
          if (neighbor_x >= 0) {
            int neighbor_color = my_IDy * Num_procsx + neighbor_x;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_right[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_right[neighbor_color]));
          }
        }

        // North Ghost
        {
          int neighbor_y = my_IDy - 1;
          if (neighbor_y >= 0) {
            int neighbor_color = neighbor_y * Num_procsx + my_IDx;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_south[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_south[neighbor_color]));
          }
        }

        // Right Ghost
        {
          int neighbor_x = my_IDx + 1;
          if (neighbor_x < Num_procsx) {
            int neighbor_color = my_IDy * Num_procsx + neighbor_x;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_left[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_left[neighbor_color]));
          }
        }

        // South Ghost
        {
          int neighbor_y = my_IDy + 1;
          if (neighbor_y < Num_procsy) {
            int neighbor_color = neighbor_y * Num_procsx + my_IDx;
            spmd_launcher.add_region_requirement(
                RegionRequirement(ghost_north[neighbor_color], READ_ONLY,
                  SIMULTANEOUS, ghost_north[neighbor_color]));
          }
        }

        spmd_launcher.add_field(0, FID_DERIV);
        spmd_launcher.add_field(0, FID_VAL);
        unsigned ghost_count = std::count (args[color].region_idx, args[color].region_idx+4, 0);
        args[color].num_regions = (ghost_count * 2) + 1 ;

        for (unsigned idx = 1; idx < args[color].num_regions; idx++){
          spmd_launcher.add_field(idx, FID_GHOST);
          if (idx <= ghost_count)
            spmd_launcher.region_requirements[idx].flags |= NO_ACCESS_FLAG;
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
    allocator.allocate_field(sizeof(DTYPE),FID_WEIGHT);
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
  init_launcher.add_arrival_barrier(args->global_barrier);
  runtime->execute_task(ctx, init_launcher);

  // Run a bunch of steps
  double ts_start = DBL_MAX, ts_end = DBL_MIN;
  for (int iter = 0; iter <= args->iterations; iter++)
  {
    if(iter == 1)
    {
      TaskLauncher dummy_launcher(TASKID_DUMMY, TaskArgument(NULL, 0));
      dummy_launcher.add_region_requirement(
          RegionRequirement(local_lr, READ_ONLY, EXCLUSIVE, local_lr));
      dummy_launcher.add_field(0, FID_DERIV);
      dummy_launcher.add_region_requirement(
          RegionRequirement(local_lr, READ_ONLY, EXCLUSIVE, local_lr));
      dummy_launcher.add_field(1, FID_VAL);
      args->global_barrier =
        runtime->advance_phase_barrier(ctx, args->global_barrier);
      dummy_launcher.add_wait_barrier(args->global_barrier);
      Future f = runtime->execute_task(ctx, dummy_launcher);
      f.get_void_result();
      ts_start = wtime();
    }

    runtime->begin_trace(ctx, 0);
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

    for (unsigned idx = 0; idx < num_neighbors; idx++)
    {
      wait_ready[idx] = runtime->advance_phase_barrier(ctx, wait_ready[idx]);
      stencil_launcher.add_wait_barrier(wait_ready[idx]);
      stencil_launcher.add_arrival_barrier(notify_empty[idx]);
      notify_empty[idx] =
        runtime->advance_phase_barrier(ctx, notify_empty[idx]);
    }
    runtime->execute_task(ctx, stencil_launcher);
    runtime->end_trace(ctx, 0);
  }

  //runtime->issue_execution_fence(ctx);
  TaskLauncher dummy_launcher(TASKID_DUMMY, TaskArgument(NULL, 0));
  dummy_launcher.add_region_requirement(
      RegionRequirement(local_lr, READ_ONLY, EXCLUSIVE, local_lr));
  dummy_launcher.add_field(0, FID_DERIV);
  dummy_launcher.add_region_requirement(
      RegionRequirement(local_lr, READ_ONLY, EXCLUSIVE, local_lr));
  dummy_launcher.add_field(1, FID_VAL);
  Future f = runtime->execute_task(ctx, dummy_launcher);
  f.get_void_result();
  ts_end = wtime();

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

  RegionAccessor<AccessorType::Generic, DTYPE> acc =
    regions[0].get_field_accessor(FID_WEIGHT).typeify<DTYPE>();

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

  SPMDArgs *args = (SPMDArgs*)task->args;

  RegionAccessor<AccessorType::Generic, DTYPE> in_acc =
    regions[0].get_field_accessor(FID_VAL).typeify<DTYPE>();
  RegionAccessor<AccessorType::Generic, DTYPE> out_acc =
    regions[1].get_field_accessor(FID_DERIV).typeify<DTYPE>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  
  DTYPE* val_ptr = 0;
  DTYPE* deriv_ptr = 0;

  int Num_procsx = args->Num_procsx;
  int Num_procsy = args->Num_procsy;
  int blockx = args->n / Num_procsx;
  int blocky = args->n / Num_procsy;
  int my_IDx = args->x;
  int my_IDy = args->y;
  int myoffsetx = my_IDx * blockx;
  int myoffsety = my_IDy * blocky;

  {
      Rect<2> subrect; ByteOffset offsets[1];
      val_ptr = in_acc.raw_rect_ptr<2>(rect, subrect, offsets);
      deriv_ptr = out_acc.raw_rect_ptr<2>(rect, subrect, offsets);
      assert(rect == subrect);
  }

#define VAL(i, j)   val_ptr[(j) * blockx + i]
#define DERIV(i, j) deriv_ptr[(j) * blockx + i]
  for (int j = 0; j < blocky; ++j)
  {
    int real_y = myoffsety + j;
    for (int i = 0; i < blockx; ++i)
    {
      int real_x = myoffsetx + i;
      DTYPE value = COEFY * real_y + COEFX * real_x;
      VAL(i, j) = value;
      DERIV(i, j) = (DTYPE)0.0;
    }
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

  RegionAccessor<AccessorType::Generic, DTYPE> write_acc =
    regions[0].get_field_accessor(write_fid).typeify<DTYPE>();
  RegionAccessor<AccessorType::Generic, DTYPE> read_acc =
    regions[1].get_field_accessor(read_fid).typeify<DTYPE>();
  RegionAccessor<AccessorType::Generic, DTYPE> weight_acc =
    regions[2].get_field_accessor(weight_fid).typeify<DTYPE>();

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

  DTYPE* write_ptr = 0;
  DTYPE* read_ptr = 0;
  DTYPE* weight_ptr = 0;
  DTYPE* left_ptr = 0;
  DTYPE* north_ptr = 0;
  DTYPE* right_ptr = 0;
  DTYPE* south_ptr = 0;

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
  RegionAccessor<AccessorType::Generic, DTYPE> left_ghost_acc;
  RegionAccessor<AccessorType::Generic, DTYPE> north_ghost_acc;
  RegionAccessor<AccessorType::Generic, DTYPE> right_ghost_acc;
  RegionAccessor<AccessorType::Generic, DTYPE> south_ghost_acc;

  unsigned idx = 0;
  if (args->region_idx[GHOST_LEFT] != -1){
      left_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<DTYPE>();
      left_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      left_rect = left_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      left_ptr = left_ghost_acc.raw_rect_ptr<2>(left_rect, subrect_g, offsets_g);
      assert(left_ptr);
      assert(left_rect == subrect_g);
      ++idx;
  }
  if (args->region_idx[GHOST_NORTH] != -1){
      north_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<DTYPE>();
      north_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      north_rect = north_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      north_ptr = north_ghost_acc.raw_rect_ptr<2>(north_rect, subrect_g, offsets_g);
      assert(north_ptr);
      assert(north_rect == subrect_g);
      ++idx;
  }
  if (args->region_idx[GHOST_RIGHT] != -1){
      right_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<DTYPE>();
      right_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      right_rect = right_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      right_ptr = right_ghost_acc.raw_rect_ptr<2>(right_rect, subrect_g, offsets_g);
      assert(right_ptr);
      assert(right_rect == subrect_g);
      ++idx;
  }
  if (args->region_idx[GHOST_SOUTH] != -1){
      south_ghost_acc = stencil_ghosts_pr[idx].get_field_accessor(ghost_fid).typeify<DTYPE>();
      south_dom = runtime->get_index_space_domain(ctx,
              stencil_ghosts_lr[idx].get_index_space());
      south_rect = south_dom.get_rect<2>();

      Rect<2> subrect_g; ByteOffset offsets_g[1];
      south_ptr = south_ghost_acc.raw_rect_ptr<2>(south_rect, subrect_g, offsets_g);
      assert(south_ptr);
      assert(south_rect == subrect_g);
  }

  int Num_procsx = args->Num_procsx;
  int Num_procsy = args->Num_procsy;
  int blockx = args->n / Num_procsx;
  int blocky = args->n / Num_procsy;
  int sizew     = 2*RADIUS+1;

#define IN(i, j)          read_ptr[(j) * blockx + i]
#define OUT(i, j)         write_ptr[(j) * blockx + i]
#define GHOST_LEFT(i, j)  left_ptr[(j) * RADIUS + i]
#define GHOST_RIGHT(i, j) right_ptr[(j) * RADIUS + i]
#define GHOST_NORTH(i, j) north_ptr[(j) * blockx + i]
#define GHOST_SOUTH(i, j) south_ptr[(j) * blockx + i]
#define WEIGHT(i, j)      weight_ptr[(j + RADIUS) * sizew + (i + RADIUS)]

  // compute the interior part
  {
    int startx = RADIUS; // inclusive
    int starty = RADIUS; // inclusive
    int endx = blockx - RADIUS; // exclusive
    int endy = blocky - RADIUS; // exclusive
    for (int j = starty; j < endy; ++j)
      for (int i = startx; i < endx; ++i)
      {
        for (int jj = -RADIUS; jj <= RADIUS; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
        for (int ii = -RADIUS; ii < 0; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (int ii = 1; ii <= RADIUS; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
      }
  }

  // compute boundaries if they exist
  if (args->region_idx[GHOST_LEFT] != -1)
  {
    int starty = RADIUS; // inclusive
    int endy = blocky - RADIUS; // exclusive
    for (int j = starty; j < endy; ++j)
      for (int i = 0; i < RADIUS; ++i)
      {
        for (int jj = -RADIUS; jj <= RADIUS; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);

        for (int ii = 1; ii <= RADIUS; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (int ii = -i; ii < 0; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (int ii = -RADIUS; ii < -i; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * GHOST_LEFT(i + RADIUS + ii, j);
      }

    if (args->region_idx[GHOST_NORTH] != -1)
      for (int j = 0; j < RADIUS; ++j)
        for (int i = 0; i < RADIUS; ++i)
        {
          for (int jj = -RADIUS; jj < -j; jj++)
            OUT(i, j) += WEIGHT(0, jj) * GHOST_NORTH(i, j + RADIUS + jj);
          for (int jj = -j; jj < 0; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
          for (int jj = 0; jj <= RADIUS; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);

          for (int ii = 1; ii <= RADIUS; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          for (int ii = -i; ii < 0; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          for (int ii = -RADIUS; ii < -i; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * GHOST_LEFT(i + RADIUS + ii, j);
        }
  }

  if (args->region_idx[GHOST_NORTH] != -1)
  {
    int startx = RADIUS; // inclusive
    int endx = blockx - RADIUS; // exclusive
    for (int j = 0; j < RADIUS; ++j)
      for (int i = startx; i < endx; ++i)
      {
        for (int jj = -RADIUS; jj < -j; jj++)
          OUT(i, j) += WEIGHT(0, jj) * GHOST_NORTH(i, j + RADIUS + jj);
        for (int jj = -j; jj < 0; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
        for (int jj = 0; jj <= RADIUS; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);

        for (int ii = -RADIUS; ii < 0; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (int ii = 1; ii <= RADIUS; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
      }

    if (args->region_idx[GHOST_RIGHT] != -1)
      for (int j = 0; j < RADIUS; ++j)
        for (int i = blockx - RADIUS; i < blockx; ++i)
        {
          for (int jj = -RADIUS; jj < -j; jj++)
            OUT(i, j) += WEIGHT(0, jj) * GHOST_NORTH(i, j + RADIUS + jj);
          for (int jj = -j; jj < 0; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
          for (int jj = 0; jj <= RADIUS; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);

          for (int ii = -RADIUS; ii < 0; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          int cutoff = blockx - i;
          for (int ii = 1; ii < cutoff; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          for (int ii = cutoff; ii <= RADIUS; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * GHOST_RIGHT(ii - cutoff, j);
        }
  }

  if (args->region_idx[GHOST_RIGHT] != -1)
  {
    int starty = RADIUS; // inclusive
    int endy = blocky - RADIUS; // exclusive
    for (int j = starty; j < endy; ++j)
      for (int i = blockx - RADIUS; i < blockx; ++i)
      {
        for (int jj = -RADIUS; jj <= RADIUS; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);

        for (int ii = -RADIUS; ii < 0; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        int cutoff = blockx - i;
        for (int ii = 1; ii < cutoff; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (int ii = cutoff; ii <= RADIUS; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * GHOST_RIGHT(ii - cutoff, j);
      }
    if (args->region_idx[GHOST_SOUTH] != -1)
      for (int j = blocky - RADIUS; j < blocky; ++j)
      {
        int cutoffy = blocky - j;
        for (int i = blockx - RADIUS; i < blockx; ++i)
        {
          for (int jj = -RADIUS; jj <= 0; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
          for (int jj = 1; jj < cutoffy; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
          for (int jj = cutoffy; jj <= RADIUS; jj++)
            OUT(i, j) += WEIGHT(0, jj) * GHOST_SOUTH(i, jj - cutoffy);

          for (int ii = -RADIUS; ii < 0; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          int cutoffx = blockx - i;
          for (int ii = 1; ii < cutoffx; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          for (int ii = cutoffx; ii <= RADIUS; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * GHOST_RIGHT(ii - cutoffx, j);
        }
      }
  }
  if (args->region_idx[GHOST_SOUTH] != -1)
  {
    int startx = RADIUS; // inclusive
    int endx = blockx - RADIUS; // exclusive
    for (int j = blocky - RADIUS; j < blocky; ++j)
    {
      int cutoff = blocky - j;
      for (int i = startx; i < endx; ++i)
      {
        for (int jj = -RADIUS; jj <= 0; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
        for (int jj = 1; jj < cutoff; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
        for (int jj = cutoff; jj <= RADIUS; jj++)
          OUT(i, j) += WEIGHT(0, jj) * GHOST_SOUTH(i, jj - cutoff);

        for (int ii = -RADIUS; ii < 0; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (int ii = 1; ii <= RADIUS; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
      }
    }

    if (args->region_idx[GHOST_LEFT] != -1)
      for (int j = blocky - RADIUS; j < blocky; ++j)
      {
        int cutoff = blocky - j;
        for (int i = 0; i < RADIUS; ++i)
        {
          for (int jj = -RADIUS; jj <= 0; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
          for (int jj = 1; jj < cutoff; jj++)
            OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
          for (int jj = cutoff; jj <= RADIUS; jj++)
            OUT(i, j) += WEIGHT(0, jj) * GHOST_SOUTH(i, jj - cutoff);

          for (int ii = 1; ii <= RADIUS; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          for (int ii = -i; ii < 0; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
          for (int ii = -RADIUS; ii < -i; ii++)
            OUT(i, j) += WEIGHT(ii, 0) * GHOST_LEFT(i + RADIUS + ii, j);
        }
      }
  }
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  SPMDArgs *args = (SPMDArgs*)task->args;

  RegionAccessor<AccessorType::Generic, DTYPE> acc =
    regions[0].get_field_accessor(FID_DERIV).typeify<DTYPE>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  DTYPE abserr = 0.0;
  DTYPE norm = 0.0;
  DTYPE value = 0.0;
  DTYPE epsilon = EPSILON; /* error tolerance */

  DTYPE* acc_ptr = 0;
  {
    Rect<2> subrect_a; ByteOffset offsets_a[1];
    acc_ptr =acc.raw_rect_ptr<2>(rect, subrect_a, offsets_a);
    assert(rect == subrect_a);
  }

  int Num_procsx = args->Num_procsx;
  int Num_procsy = args->Num_procsy;
  int blockx = args->n / Num_procsx;
  int blocky = args->n / Num_procsy;
  int my_IDx = args->x;
  int my_IDy = args->y;
  int myoffsetx = my_IDx * blockx;
  int myoffsety = my_IDy * blocky;

  int startx = 0;
  int starty = 0;
  int endx = blockx-1;
  int endy = blocky-1;

  if(my_IDx == 0)        startx += RADIUS;
  if(my_IDy == 0)        starty += RADIUS;
  if(my_IDx == Num_procsx-1) endx   -= RADIUS;
  if(my_IDy == Num_procsy-1) endy   -= RADIUS;

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
    if (args->my_ID == 0) printf("Solution validates\n");
#ifdef VERBOSE
    if (args->my_ID == 0) printf("Squared errors: %f \n", abserr);
#endif
  }
  else {
    if (args->my_ID == 0) printf("ERROR: Squared error %lf exceeds threshold %e\n",
        abserr, epsilon);
    exit(EXIT_FAILURE);
  }
}

static void register_mappers(Machine machine, Runtime *rt,
                             const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
      it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new StencilMapper(machine, rt, *it), *it);
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
      AUTO_GENERATE_ID, TaskConfigOptions(false, true), "spmd");
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

  HighLevelRuntime::set_registration_callback(register_mappers);
  return HighLevelRuntime::start(argc, argv);
}
