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
         Completely rewritten by Wonchan Lee, Mar 2016.

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
    virtual void select_task_options(Task *task);
    virtual void slice_domain(const Task *task, const Domain &domain,
        std::vector<DomainSplit> &slices);
    virtual bool map_task(Task* task);
    virtual bool map_must_epoch(const std::vector<Task*> &tasks,
        const std::vector<MappingConstraint> &constraints,
        MappingTagID tag);
  private:
    std::map<AddressSpaceID, std::vector<Processor> > proc_map;
    std::map<Processor, Memory> all_sysmems;
    int num_nodes;
};

StencilMapper::StencilMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local), num_nodes(0)
{
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  machine_interface.filter_processors(machine, Processor::LOC_PROC, all_procs);

  for (std::set<Processor>::iterator itr = all_procs.begin();
      itr != all_procs.end(); ++itr)
  {
    int node_id = itr->address_space();
    proc_map[node_id].push_back(*itr);
    Memory sysmem = machine_interface.find_memory_kind(*itr, Memory::SYSTEM_MEM);
    assert(sysmem.exists());
    all_sysmems[*itr] = sysmem;
    num_nodes = std::max(num_nodes, node_id + 1);
  }
}

void StencilMapper::slice_domain(const Task *task, const Domain &domain,
                                 std::vector<DomainSplit> &slices)
{
  std::vector<Processor>& procs =
    proc_map[task->target_proc.address_space()];
  DefaultMapper::decompose_index_space(domain, procs,
      1, slices);
}

void StencilMapper::select_task_options(Task *task)
{
  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = true;
  task->profile_task = false;
  task->task_priority = 0;
}

bool StencilMapper::map_task(Task *task)
{
  Memory sysmem = all_sysmems[task->target_proc];
  std::vector<RegionRequirement> &regions = task->regions;
  for (unsigned idx = 0; idx < regions.size(); ++idx)
  {
    RegionRequirement &req = regions[idx];

    // Region options:
    req.virtual_map = false;
    req.enable_WAR_optimization = false;
    req.reduction_list = false;

    req.blocking_factor = req.max_blocking_factor;
    req.target_ranking.push_back(sysmem);
  }
  return false;
}

bool StencilMapper::map_must_epoch(const std::vector<Task*> &tasks,
    const std::vector<MappingConstraint> &constraints,
    MappingTagID tag)
{
  for (unsigned i = 0; i < tasks.size(); ++i)
  {
    Task* task = tasks[i];
    task->target_proc = *proc_map[i].begin();
    map_task(task);
  }

  typedef std::map<LogicalRegion, Memory> Mapping;
  Mapping mappings;
  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    assert(c.t1->target_proc.address_space() !=
           c.t2->target_proc.address_space());
    if (c.idx1 == 0)
    {
      Memory sysmem = all_sysmems[c.t1->target_proc];
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(sysmem);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(sysmem);
      mappings[c.t1->regions[c.idx1].region] = sysmem;
    }
    else if (c.idx2 == 0)
    {
      Memory sysmem = all_sysmems[c.t2->target_proc];
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(sysmem);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(sysmem);
      mappings[c.t2->regions[c.idx2].region] = sysmem;
    }
    else
      continue;
  }

  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    if (c.idx1 != 0 && c.idx2 != 0)
    {
      Mapping::iterator it = mappings.find(c.t1->regions[c.idx1].region);
      assert(it != mappings.end());
      Memory regmem = it->second;
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(regmem);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(regmem);
    }
  }

  //for (unsigned i = 0; i < tasks.size(); ++i)
  //{
  //  Task* task = tasks[i];
  //  printf("processor " IDFMT "\n", task->target_proc);
  //  for (unsigned j = 0; j < task->regions.size(); ++j)
  //    printf("  region %d memory " IDFMT "\n",
  //        j, task->regions[j].target_ranking.begin()->id);
  //}

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
  FID_IN,
  FID_OUT,
  FID_WEIGHT,
};

enum Direction {
  GHOST_LEFT = 0,
  GHOST_UP,
  GHOST_RIGHT,
  GHOST_DOWN,
  PRIVATE,
};

struct SPMDArgs {
  int n;
  int numThreads;
  int numIterations;
  int myRank;
  Point<2> tileLoc;
  PhaseBarrier inputReady;
};

struct StencilArgs {
  int n;
  int numIterations;
  int haloX;
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
  int Num_procsx, Num_procsy, num_ranks, threads;
  int iterations;

  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/
  const InputArgs &inputs = HighLevelRuntime::get_input_args();

  if (inputs.argc < 4)
  {
    printf("Usage: %s <# threads> <# iterations> <array dimension>\n",
           *inputs.argv);
    exit(EXIT_FAILURE);
  }

  num_ranks = gasnet_nodes();

  threads = atoi((inputs.argv[1]));
  if (threads <= 0)
  {
      printf("ERROR: Number of THREADS must be > 0 : %d \n", threads);
      exit(EXIT_FAILURE);
  }

  iterations = atoi((inputs.argv[2]));
  if (iterations < 1)
  {
      printf("ERROR: iterations must be >= 1 : %d \n", iterations);
      exit(EXIT_FAILURE);
  }

  n = atoi(inputs.argv[3]);
  if (n <= 0)
  {
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

  /* compute "processor" grid; initialize Num_procsy to avoid compiler warnings */
  Num_procsy = 0;
  for (Num_procsx = (int)(sqrt(num_ranks + 1));
       Num_procsx > 0; --Num_procsx)
    if (num_ranks % Num_procsx == 0)
    {
      Num_procsy = num_ranks / Num_procsx;
      break;
    }
  //std::swap(Num_procsx, Num_procsy);

  if (n % num_ranks != 0)
  {
    printf("n%%#ranks should be zero\n");
    exit(EXIT_FAILURE);
  }
  if (n % Num_procsx != 0)
  {
    printf("n%%Num_procsx should be zero\n");
    exit(EXIT_FAILURE);
  }

  if (n % Num_procsy != 0)
  {
    printf("n%%Num_procsy should be zero\n");
    exit(EXIT_FAILURE);
  }

  if (RADIUS < 1)
  {
    printf("Stencil radius %d should be positive\n", RADIUS);
    exit(EXIT_FAILURE);
  }

  if (2 * RADIUS + 1 > n)
  {
    printf("Stencil radius %d exceeds grid size %d\n", RADIUS, n);
    exit(EXIT_FAILURE);
  }

  if ((n / num_ranks) % threads != 0)
  {
    printf("(n / num_ranks)%%THREADS should be zero\n");
    exit(EXIT_FAILURE);
  }

  printf("Tiles in x/y-direction = %d/%d\n", Num_procsx, Num_procsy);

  /*********************************************************************
  ** Create the master index space
  *********************************************************************/
  Domain domain =
    Domain::from_rect<2>(Rect<2>(make_point(0, 0), make_point(n - 1, n - 1)));
  IndexSpace is = runtime->create_index_space(ctx, domain);

  /*********************************************************************
  ** Create a partition for tiles
  *********************************************************************/
  Domain colorSpace =
    Domain::from_rect<2>(
        Rect<2>(make_point(0, 0), make_point(Num_procsx - 1, Num_procsy - 1)));

  int tileSizeX = n / Num_procsx;
  int tileSizeY = n / Num_procsy;

  DomainPointColoring haloColoring;

  for (int tileY = 0; tileY < Num_procsy; ++tileY)
    for (int tileX = 0; tileX < Num_procsx; ++tileX)
    {
      DomainPoint tilePoint =
        DomainPoint::from_point<2>(make_point(tileX, tileY));

      Domain tileDomain = Domain::from_rect<2>(Rect<2>(
            make_point(
              std::max(tileX * tileSizeX - RADIUS, 0),
              std::max(tileY * tileSizeY - RADIUS, 0)),
            make_point(
              std::min((tileX + 1) * tileSizeX + RADIUS, n) - 1,
              std::min((tileY + 1) * tileSizeY + RADIUS, n) - 1)));

      haloColoring[tilePoint] = tileDomain;
    }

  IndexPartition haloIp =
    runtime->create_index_partition(ctx, is, colorSpace, haloColoring);

  /*********************************************************************
  ** Create top-level regions
  *********************************************************************/
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(DTYPE), FID_IN);
    allocator.allocate_field(sizeof(DTYPE), FID_OUT);
  }
  std::map<DomainPoint, LogicalRegion> privateLrs;
  for (int tileY = 0; tileY < Num_procsy; ++tileY)
    for (int tileX = 0; tileX < Num_procsx; ++tileX)
    {
      DomainPoint tilePoint =
        DomainPoint::from_point<2>(make_point(tileX, tileY));

      IndexSpace subspace =
        runtime->get_index_subspace(ctx, haloIp, tilePoint);
      LogicalRegion privateLr =
        runtime->create_logical_region(ctx, subspace, fs);
      privateLrs[tilePoint] = privateLr;
    }

  /*********************************************************************
  ** Launch SPMD shards
  *********************************************************************/
  PhaseBarrier inputReady = runtime->create_phase_barrier(ctx, num_ranks);
  int offsets[][2] = { {-1, 0}, {0, -1}, {1, 0}, {0, 1} };
  MustEpochLauncher shardLauncher;
  std::map<DomainPoint, SPMDArgs> args;
  for (int tileY = 0; tileY < Num_procsy; ++tileY)
    for (int tileX = 0; tileX < Num_procsx; ++tileX)
    {
      DomainPoint tilePoint =
        DomainPoint::from_point<2>(make_point(tileX, tileY));

      args[tilePoint].n = n;
      args[tilePoint].numThreads = threads;
      args[tilePoint].numIterations = iterations;
      args[tilePoint].tileLoc = tilePoint.get_point<2>();
      args[tilePoint].inputReady = inputReady;

      TaskLauncher spmdLauncher(TASKID_SPMD,
          TaskArgument(&args[tilePoint], sizeof(SPMDArgs)));
      RegionRequirement req(privateLrs[tilePoint], READ_WRITE,
          SIMULTANEOUS, privateLrs[tilePoint]);
      req.add_field(FID_IN);
      req.add_field(FID_OUT);
      spmdLauncher.add_region_requirement(req);
      for (int dir = GHOST_LEFT; dir <= GHOST_DOWN; ++dir)
      {
        coord_t neighborCoords[] = {
          tileX + offsets[dir][0],
          tileY + offsets[dir][1]
        };
        if (neighborCoords[0] < 0 || neighborCoords[1] < 0 ||
            neighborCoords[0] >= Num_procsx || neighborCoords[1] >= Num_procsy)
          continue;
        DomainPoint neighborPoint =
          DomainPoint::from_point<2>(Point<2>(neighborCoords));
        RegionRequirement req(privateLrs[neighborPoint], READ_WRITE,
            SIMULTANEOUS, privateLrs[neighborPoint]);
        req.add_field(FID_IN);
        req.add_field(FID_OUT);
        req.flags |= NO_ACCESS_FLAG;
        spmdLauncher.add_region_requirement(req);
      }

      shardLauncher.add_single_task(tilePoint, spmdLauncher);
    }

  FutureMap fm = runtime->execute_must_epoch(ctx, shardLauncher);
  fm.wait_all_results();

#ifdef WALL_CLOCK_TIME
  double tsStart = DBL_MAX, tsEnd = DBL_MIN;
#else
  double maxTime = DBL_MIN;
#endif
  for (int tileY = 0; tileY < Num_procsy; ++tileY)
    for (int tileX = 0; tileX < Num_procsx; ++tileX)
    {
      DomainPoint tilePoint =
        DomainPoint::from_point<2>(make_point(tileX, tileY));
      typedef std::pair<double, double> Doubles;
      Doubles p = fm.get_result<Doubles>(tilePoint);
#ifdef WALL_CLOCK_TIME
      tsStart = MIN(tsStart, p.first);
      tsEnd = MAX(tsEnd, p.second);
#else
      maxTime = MAX(maxTime, p.second - p.first);
#endif
    }

#ifdef WALL_CLOCK_TIME
  double maxTime = tsEnd - tsStart;
#endif
  double avgTime = maxTime / iterations;

  int stencil_size = 4 * RADIUS + 1;
  unsigned long active_points =
    (unsigned long)(n - 2 * RADIUS) * (n - 2 * RADIUS);
  double flops = (DTYPE) (2 * stencil_size + 1) * active_points;

  printf("Rate (MFlops/s): " FSTR "  Avg time (s): %lf\n",
      1.0E-06 * flops / avgTime, avgTime);
}

static LogicalPartition createHaloPartition(LogicalRegion lr,
                                            int n,
                                            Context ctx,
                                            HighLevelRuntime *runtime)
{
  IndexSpace is = lr.get_index_space();
  Rect<2> haloBox =
    runtime->get_index_space_domain(ctx, is).get_rect<2>();
  Rect<2> boundingBox = haloBox;
  for (int i = 0; i < 2; ++i)
  {
    if (boundingBox.lo[i] != 0) boundingBox.lo.x[i] += RADIUS;
    if (boundingBox.hi[i] != n - 1) boundingBox.hi.x[i] -= RADIUS;
  }

  Domain colorSpace = Domain::from_rect<1>(Rect<1>(GHOST_LEFT, PRIVATE));
  DomainPointColoring coloring;
  std::vector<DomainPoint> colors;
  for (int color = GHOST_LEFT; color <= PRIVATE; ++color)
    colors.push_back(DomainPoint::from_point<1>(color));

  if (boundingBox.lo[0] > 0)
  {
    coord_t lu[] = { haloBox.lo[0], boundingBox.lo[1] };
    coord_t rd[] = { boundingBox.lo[0] - 1, boundingBox.hi[1] };
    coloring[colors[GHOST_LEFT]] =
      Domain::from_rect<2>(Rect<2>(Point<2>(lu), Point<2>(rd)));
  }
  if (boundingBox.lo[1] > 0)
  {
    coord_t lu[] = { boundingBox.lo[0], haloBox.lo[1] };
    coord_t rd[] = { boundingBox.hi[0], boundingBox.lo[1] - 1 };
    coloring[colors[GHOST_UP]] =
      Domain::from_rect<2>(Rect<2>(Point<2>(lu), Point<2>(rd)));
  }
  if (boundingBox.hi[0] < n - 1)
  {
    coord_t lu[] = { boundingBox.hi[0] + 1, boundingBox.lo[1] };
    coord_t rd[] = { haloBox.hi[0], boundingBox.hi[1] };
    coloring[colors[GHOST_RIGHT]] =
      Domain::from_rect<2>(Rect<2>(Point<2>(lu), Point<2>(rd)));
  }
  if (boundingBox.hi[1] < n - 1)
  {
    coord_t lu[] = { boundingBox.lo[0], boundingBox.hi[1] + 1 };
    coord_t rd[] = { boundingBox.hi[0], haloBox.hi[1] };
    coloring[colors[GHOST_DOWN]] =
      Domain::from_rect<2>(Rect<2>(Point<2>(lu), Point<2>(rd)));
  }

  coloring[colors[PRIVATE]] = Domain::from_rect<2>(boundingBox);
  IndexPartition ip =
    runtime->create_index_partition(ctx, is, colorSpace, coloring,
                                    DISJOINT_KIND);

  return runtime->get_logical_partition(ctx, lr, ip);
}

std::pair<double, double> spmd_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime)
{
  SPMDArgs *args = (SPMDArgs*)task->args;
  int n = args->n;
  int numThreads = args->numThreads;
  Point<2> tileLoc = args->tileLoc;
  PhaseBarrier inputReady = args->inputReady;

  LogicalRegion localLr = regions[0].get_logical_region();
  LogicalPartition localLP = createHaloPartition(localLr, n, ctx, runtime);

  IndexSpace is = localLr.get_index_space();
  Rect<2> haloBox = runtime->get_index_space_domain(ctx, is).get_rect<2>();
  Rect<2> boundingBox = haloBox;
  for (int i = 0; i < 2; ++i)
  {
    if (boundingBox.lo[i] != 0) boundingBox.lo.x[i] += RADIUS;
    if (boundingBox.hi[i] != n - 1) boundingBox.hi.x[i] -= RADIUS;
  }

  std::vector<bool> hasNeighbor(4);
  hasNeighbor[GHOST_LEFT] = haloBox.lo[0] != 0;
  hasNeighbor[GHOST_UP] = haloBox.lo[1] != 0;
  hasNeighbor[GHOST_RIGHT] = haloBox.hi[0] != n - 1;
  hasNeighbor[GHOST_DOWN] = haloBox.hi[1] != n - 1;

  // create local partition
  LogicalRegion privateLr =
      runtime->get_logical_subregion_by_color(ctx, localLP,
          DomainPoint::from_point<1>(PRIVATE));
  // TODO: handle uneven local partition
  IndexSpace privateIs = privateLr.get_index_space();
  int blockX = boundingBox.hi[0] - boundingBox.lo[0] + 1;
  int blockY = boundingBox.hi[1] - boundingBox.lo[1] + 1;
  assert(blockY % numThreads == 0);

  IndexPartition privateIp =
    runtime->create_index_partition(ctx, privateIs,
        Blockify<2>(make_point(blockX, blockY / numThreads)));
  LogicalPartition privateLp =
    runtime->get_logical_partition(ctx, privateLr, privateIp);

  // get neighbors' logical region
  std::vector<LogicalRegion> neighborLrs(4);
  unsigned idx = 1;
  for (unsigned dir = GHOST_LEFT; dir <= GHOST_DOWN; ++dir)
    if (hasNeighbor[dir])
      neighborLrs[dir] = regions[idx++].get_logical_region();

  std::vector<LogicalRegion> ghostLrs(4);
  std::vector<LogicalRegion> bufferLrs(4);
  for (unsigned dir = GHOST_LEFT; dir <= GHOST_DOWN; ++dir)
  {
    LogicalRegion ghostLr =
      runtime->get_logical_subregion_by_color(ctx, localLP,
          DomainPoint::from_point<1>(dir));
    ghostLrs[dir] = ghostLr;
    bufferLrs[dir] =
      runtime->create_logical_region(ctx, ghostLr.get_index_space(),
          ghostLr.get_field_space());
  }

  Domain launchDomain = Domain::from_rect<2>(Rect<2>(
        make_point(tileLoc[0], numThreads * tileLoc[1]),
        make_point(tileLoc[0], numThreads * (tileLoc[1] + 1) - 1)));

  // setup arguments for the child tasks
  StencilArgs stencilArgs;
  stencilArgs.n = n;
  stencilArgs.numIterations = args->numIterations;
  stencilArgs.haloX = haloBox.hi[0] - haloBox.lo[0] + 1;
  TaskArgument taskArg(&stencilArgs, sizeof(StencilArgs));

  // initialize fields
  ArgumentMap argMap;
  {
    IndexLauncher initLauncher(TASKID_INITIALIZE, launchDomain,
        taskArg, argMap);
    RegionRequirement req(privateLp, 0, READ_WRITE, EXCLUSIVE, localLr);
    req.add_field(FID_IN);
    req.add_field(FID_OUT);
    initLauncher.add_region_requirement(req);
    initLauncher.add_arrival_barrier(inputReady);
    inputReady = runtime->advance_phase_barrier(ctx, inputReady);
    FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
    fm.wait_all_results();
  }

  // create a logical region for weights
  LogicalRegion weightLr;

  {
    Domain domain = Domain::from_rect<2>(Rect<2>(
          make_point(-RADIUS, -RADIUS),
          make_point(RADIUS, RADIUS)));
    IndexSpace is = runtime->create_index_space(ctx, domain);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(DTYPE), FID_WEIGHT);
    }
    weightLr = runtime->create_logical_region(ctx, is, fs);
  }

  // initialize weights
  {
    TaskLauncher weightInitLauncher(TASKID_WEIGHT_INITIALIZE,
        TaskArgument(0, 0));
    RegionRequirement req(weightLr, WRITE_DISCARD, EXCLUSIVE, weightLr);
    req.add_field(FID_WEIGHT);
    weightInitLauncher.add_region_requirement(req);
    Future f = runtime->execute_task(ctx, weightInitLauncher);
    f.get_void_result();
  }

  {
    TaskLauncher dummyLauncher(TASKID_DUMMY, TaskArgument(0, 0));
    dummyLauncher.add_region_requirement(
        RegionRequirement(localLr, READ_ONLY, EXCLUSIVE, localLr));
    dummyLauncher.add_field(0, FID_IN);
    dummyLauncher.add_field(0, FID_OUT);
    dummyLauncher.add_wait_barrier(inputReady);
    Future f = runtime->execute_task(ctx, dummyLauncher);
    f.get_void_result();
  }

  double tsStart = wtime();

  FutureMap fm;
  for (int iter = 0; iter < stencilArgs.numIterations; iter++)
  {
    runtime->begin_trace(ctx, 0);
    for (unsigned dir = GHOST_LEFT; dir <= GHOST_DOWN; ++dir)
      if (hasNeighbor[dir])
      {
        {
          RegionRequirement srcReq(neighborLrs[dir], READ_ONLY, EXCLUSIVE,
                                   neighborLrs[dir]);
          srcReq.add_field(FID_IN);
          RegionRequirement dstReq(bufferLrs[dir], READ_WRITE, EXCLUSIVE,
                                   bufferLrs[dir]);
          dstReq.add_field(FID_IN);

          CopyLauncher copyLauncher;
          copyLauncher.add_copy_requirements(srcReq, dstReq);
          runtime->issue_copy_operation(ctx, copyLauncher);
        }

        {
          RegionRequirement srcReq(bufferLrs[dir], READ_ONLY, EXCLUSIVE,
                                   bufferLrs[dir]);
          srcReq.add_field(FID_IN);
          RegionRequirement dstReq(ghostLrs[dir], READ_WRITE, EXCLUSIVE,
                                   localLr);
          dstReq.add_field(FID_IN);

          CopyLauncher copyLauncher;
          copyLauncher.add_copy_requirements(srcReq, dstReq);
          runtime->issue_copy_operation(ctx, copyLauncher);
        }
      }

    {
      IndexLauncher stencilLauncher(TASKID_STENCIL, launchDomain, taskArg,
          argMap);
      RegionRequirement req(privateLp, 0, READ_ONLY, EXCLUSIVE, localLr);
      req.add_field(FID_IN);
      req.add_field(FID_OUT);
      RegionRequirement weightReq(weightLr, READ_ONLY, EXCLUSIVE, weightLr);
      weightReq.add_field(FID_WEIGHT);
      // need this requirement to block copies
      RegionRequirement dummyReq(localLr, 0, READ_ONLY, EXCLUSIVE, localLr);
      dummyReq.add_field(FID_IN);
      stencilLauncher.add_region_requirement(req);
      stencilLauncher.add_region_requirement(weightReq);
      stencilLauncher.add_region_requirement(dummyReq);
      fm = runtime->execute_index_space(ctx, stencilLauncher);
    }
    runtime->end_trace(ctx, 0);
  }
  fm.wait_all_results();

  double tsEnd = wtime();

  {
    IndexLauncher checkLauncher(TASKID_CHECK, launchDomain, taskArg, argMap);
    RegionRequirement req(privateLp, 0, READ_ONLY, EXCLUSIVE, localLr);
    req.add_field(FID_OUT);
    checkLauncher.add_region_requirement(req);
    FutureMap fm = runtime->execute_index_space(ctx, checkLauncher);
    fm.wait_all_results();

    DTYPE abserr = 0.0;
    DTYPE epsilon = EPSILON; /* error tolerance */

    for (Domain::DomainPointIterator it(launchDomain); it; it++)
      abserr += fm.get_result<double>(it.p);

    if (abserr < epsilon)
    {
      if (args->tileLoc[0] == 0 && args->tileLoc[1] == 0)
        printf("Solution validates\n");
#ifdef VERBOSE
      if (args->tileLoc[0] == 0 && args->tileLoc[1] == 0)
        printf("Squared errors: %f \n", abserr);
#endif
    }
    else
    {
      fprintf(stderr,
          "ERROR: Squared error %lf exceeds threshold %e (%ld, %ld)\n",
          abserr, epsilon, args->tileLoc[0], args->tileLoc[1]);
      exit(EXIT_FAILURE);
    }
  }

  return std::pair<double, double>(tsStart, tsEnd);
}

void dummy_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
}

void init_weight_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  RegionAccessor<AccessorType::Generic, DTYPE> acc =
    regions[0].get_field_accessor(FID_WEIGHT).typeify<DTYPE>();
  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    Point<2> p = pir.p;
    coord_t xx = p[0];
    coord_t yy = p[1];

    if (yy == 0)
    {
      if (xx == 0) continue;
      else
      {
        DTYPE val = (1.0 / (2.0 * xx * RADIUS));
        acc.write(DomainPoint::from_point<2>(p), val);
      }
    }
    else if (xx == 0)
    {
      if (yy == 0) continue;
      else
      {
        DTYPE val = (1.0 / (2.0* yy * RADIUS));
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
  RegionAccessor<AccessorType::Generic, DTYPE> inputAcc =
    regions[0].get_field_accessor(FID_IN).typeify<DTYPE>();
  RegionAccessor<AccessorType::Generic, DTYPE> outputAcc =
    regions[0].get_field_accessor(FID_OUT).typeify<DTYPE>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  StencilArgs *args = (StencilArgs*)task->args;
  coord_t luX = rect.lo[0];
  coord_t luY = rect.lo[1];
  coord_t blockX = rect.hi[0] - luX + 1;
  coord_t blockY = rect.hi[1] - luY + 1;
  coord_t haloX = args->haloX;

  DTYPE* inPtr = 0;
  DTYPE* outPtr = 0;
  {
    Rect<2> s; ByteOffset bo[1];
    inPtr = inputAcc.raw_rect_ptr<2>(rect, s, bo);
    outPtr = outputAcc.raw_rect_ptr<2>(rect, s, bo);
  }

#define IN(i, j)   inPtr[(j) * haloX + i]
#define OUT(i, j) outPtr[(j) * haloX + i]
  for (coord_t j = 0; j < blockY; ++j)
  {
    coord_t realY = luY + j;
    for (coord_t i = 0; i < blockX; ++i)
    {
      coord_t realX = luX + i;
      DTYPE value = COEFY * realY + COEFX * realX;
      IN(i, j) = value;
      OUT(i, j) = (DTYPE)0.0;
    }
  }
#undef IN
#undef OUT
}

void stencil(DTYPE* RESTRICT inputPtr,
             DTYPE* RESTRICT outputPtr,
             DTYPE* RESTRICT weightPtr,
             coord_t haloX, coord_t startX, coord_t endX,
             coord_t startY, coord_t endY)
{
#define IN(i, j)     inputPtr[(j) * haloX + i]
#define OUT(i, j)    outputPtr[(j) * haloX + i]
#define WEIGHT(i, j) weightPtr[(j + RADIUS) * (2 * RADIUS + 1) + (i + RADIUS)]

  coord_t i, j, ii, jj;

  for (j = startY; j < endY; ++j)
    for (i = startX; i < endX; ++i)
    {
      for (jj = -RADIUS; jj <= RADIUS; jj++)
        OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
      for (ii = -RADIUS; ii < 0; ii++)
        OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
      for (ii = 1; ii <= RADIUS; ii++)
        OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
    }

#undef IN
#undef OUT
#undef WEIGHT
}

void stencil_field_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  RegionAccessor<AccessorType::Generic, DTYPE> inputAcc =
    regions[0].get_field_accessor(FID_IN).typeify<DTYPE>();
  RegionAccessor<AccessorType::Generic, DTYPE> outputAcc =
    regions[0].get_field_accessor(FID_OUT).typeify<DTYPE>();
  RegionAccessor<AccessorType::Generic, DTYPE> weightAcc =
    regions[1].get_field_accessor(FID_WEIGHT).typeify<DTYPE>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain weightDom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  Rect<2> weightRect = weightDom.get_rect<2>();

  // get raw pointers
  DTYPE* inputPtr = 0;
  DTYPE* outputPtr = 0;
  DTYPE* weightPtr = 0;
  {
      Rect<2> r; ByteOffset bo[1];
      inputPtr = inputAcc.raw_rect_ptr<2>(rect, r, bo);
      outputPtr = outputAcc.raw_rect_ptr<2>(rect, r, bo);
      weightPtr = weightAcc.raw_rect_ptr<2>(weightRect, r, bo);
  }

  StencilArgs *args = (StencilArgs*)task->args;
  int n = args->n;
  coord_t haloX = args->haloX;
  coord_t luX = rect.lo[0];
  coord_t luY = rect.lo[1];
  coord_t rdX = rect.hi[0];
  coord_t rdY = rect.hi[1];
  coord_t startX = 0;
  coord_t startY = 0;

  if (luX == 0) { luX += RADIUS; startX += RADIUS; }
  if (luY == 0) { luY += RADIUS; startY += RADIUS; }
  if (rdX == n - 1) rdX -= RADIUS;
  if (rdY == n - 1) rdY -= RADIUS;
  coord_t endX = startX + (rdX - luX + 1);
  coord_t endY = startY + (rdY - luY + 1);

  stencil(inputPtr, outputPtr, weightPtr, haloX, startX, endX, startY, endY);
}

double check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  RegionAccessor<AccessorType::Generic, DTYPE> acc =
    regions[0].get_field_accessor(FID_OUT).typeify<DTYPE>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  StencilArgs *args = (StencilArgs*)task->args;
  int n = args->n;
  coord_t luX = rect.lo[0];
  coord_t luY = rect.lo[1];
  coord_t blockX = rect.hi[0] - luX + 1;
  coord_t blockY = rect.hi[1] - luY + 1;
  coord_t haloX = args->haloX;

  DTYPE* ptr = 0;
  {
    Rect<2> s; ByteOffset bo[1];
    ptr = acc.raw_rect_ptr<2>(rect, s, bo);
  }

  DTYPE abserr = 0.0;

#define OUT(i, j) ptr[(j) * haloX + i]
  for (coord_t j = 0; j < blockY; ++j)
  {
    coord_t realY = luY + j;
    for (coord_t i = 0; i < blockX; ++i)
    {
      coord_t realX = luX + i;

      if (realX < RADIUS || realY < RADIUS) continue;
      if (realX >= n - RADIUS || realY >= n - RADIUS) continue;

      DTYPE norm = (DTYPE) (args->numIterations) * (COEFX + COEFY);
      DTYPE value = OUT(i, j);
      abserr += ABS(value - norm);
    }
  }
#undef OUT

  return abserr;
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
      AUTO_GENERATE_ID, TaskConfigOptions(false, true), "top_level");
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
      AUTO_GENERATE_ID, TaskConfigOptions(true), "dummy");
  HighLevelRuntime::register_legion_task<double, check_task>(TASKID_CHECK,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "check");

  HighLevelRuntime::set_registration_callback(register_mappers);
  return HighLevelRuntime::start(argc, argv);
}
