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

NAME:    transpose

PURPOSE: This program tests the efficiency with which a square matrix
         can be transposed and stored in another matrix. The matrices
         are distributed identically.

USAGE:   Program input is three command line arguments that give the
         matrix order, the number of times to repeat the operation
         (iterations), and the number of threads to use:

         <progname> <# threads> <matrix_size> <# iterations> <tile size>

         An optional parameter specifies the tile size used to divide the
         individual matrix blocks for improved cache and TLB performance.

         The output consists of diagnostics to make sure the
         transpose worked and timing statistics.

HISTORY: Written by Abdullah Kayi, May 2015.

*******************************************************************/
#include "../../include/par-res-kern_legion.h"

#include <sys/time.h>
#define  USEC_TO_SEC   1.0e-6    /* to convert microsecs to secs */
#include "default_mapper.h"

enum TaskIDs {
  TASKID_TOPLEVEL = 1,
  TASKID_INITIALIZE_A,
  TASKID_INITIALIZE_B,
  TASKID_TRANSPOSE,
  TASKID_CHECK,
};

enum FieldIDs {
  FID_X = 1,
  FID_Y
};

struct BlockArgs {
  int order;
  int tile_size;
  int iterations;
  int num_partitions;
};

double wtime() {
  double time_seconds;

  struct timeval  time_data; /* seconds since 0 GMT */

  gettimeofday(&time_data,NULL);

  time_seconds  = (double) time_data.tv_sec;
  time_seconds += (double) time_data.tv_usec * USEC_TO_SEC;

  return time_seconds;
}

void task_toplevel(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, HighLevelRuntime *runtime)
{
  int num_partitions, order, tile_size, iterations;

  const InputArgs &inputs = HighLevelRuntime::get_input_args();

  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/

  if (inputs.argc < 5){
    printf("Usage: %s  <# threads> <# iterations> <matrix order> <# partitions>[tile size]\n",
           *inputs.argv);
    exit(EXIT_FAILURE);
  }

  num_partitions = atoi((inputs.argv[1]));
  if (num_partitions < 1){
      printf("ERROR: partitions must be >= 1 : %d \n", num_partitions);
      exit(EXIT_FAILURE);
  }

  iterations  = atoi((inputs.argv[2]));
  if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      exit(EXIT_FAILURE);
  }

  order = atoi(inputs.argv[3]);
  if (order < 0){
      printf("ERROR: Matrix Order must be greater than 0 : %d \n", order);
      exit(EXIT_FAILURE);
  }

  if (inputs.argc >= 5) {
      tile_size = atoi(inputs.argv[4]);
  } else {
      tile_size = order;
  }
  /* a non-positive tile size means no tiling of the local transpose */
  if (tile_size <=0) tile_size = order;

  printf("Parallel Research Kernels version %s\n", PRKVERSION);
  printf("Legion matrix transpose: B = A^T\n");
  printf("Number of threads    = %d\n", num_partitions);
  printf("Matrix order         = %d\n", order);
  printf("Number of iterations = %d\n", iterations);
  if ((tile_size > 0) && (tile_size < order))
          printf("Tile size            = %d\n", tile_size);
  else    printf("Untiled\n");

  BlockArgs args;
  args.order = order;
  args.tile_size = tile_size;
  args.iterations = iterations;
  args.num_partitions = num_partitions;

  /*********************************************************************
  ** Create index/field space and logical regions for the input/output matrix
  *********************************************************************/

  // Using the same 1-D approach in the serial code to contain 2-D matrix
  Rect<1> elem_rect(Point<1>(0),Point<1>(order*order-1));
  Domain domain = Domain::from_rect<1>(elem_rect);
  IndexSpace is = runtime->create_index_space(ctx, domain);

  FieldSpace ifs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, ifs);
    allocator.allocate_field(sizeof(double),FID_X);
  }

  FieldSpace ofs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, ofs);
    allocator.allocate_field(sizeof(double),FID_Y);
  }

  // Two logical regions with same index and field space
  // holding the logical region for input and output matrices
  LogicalRegion lr_a = runtime->create_logical_region(ctx, is, ifs);
  LogicalRegion lr_b = runtime->create_logical_region(ctx, is, ofs);

  /*********************************************************************
  ** Create partitions for data level parallelism
  *********************************************************************/
  // Each partition will be associated with a unique color(unsigned int)
  // Here Rect and Domain are used to depict our partitioning scheme
  Rect<1> color_bounds(Point<1>(0),Point<1>(num_partitions-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  IndexPartition ip;
  // Assuming evenly partitioned blocks here
  // Blockify will evenly divide a rectangle into implicitly disjoint subsets
  // containing the specified number of points in each block.
  Blockify<1> coloring(order * order/num_partitions);
  ip = runtime->create_index_partition(ctx, is, coloring);

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, lr_a, ip);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, lr_b, ip);

  /*********************************************************************
  ** Launch sub-tasks for initialization
  *********************************************************************/
  // Create our launch domain same as color domain as we are going to
  // launch one task for each partition
  Domain launch_domain = color_domain;
  ArgumentMap arg_map;

  IndexLauncher init_a_launcher(TASKID_INITIALIZE_A, launch_domain,
                                TaskArgument(NULL, 0), arg_map);
  init_a_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0 /*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, lr_a));
  init_a_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, init_a_launcher);

  IndexLauncher init_b_launcher(TASKID_INITIALIZE_B, launch_domain,
                                TaskArgument(NULL, 0), arg_map);
  init_b_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0 /*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, lr_b));
  init_b_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, init_b_launcher);

  /*********************************************************************
  ** Launch sub-task for transpose
  *********************************************************************/

  IndexLauncher transpose_launcher(TASKID_TRANSPOSE, launch_domain,
      TaskArgument(&args, sizeof(args)), arg_map);
  transpose_launcher.add_region_requirement(
      RegionRequirement(lr_a, 0/*projection ID*/,
        READ_ONLY, EXCLUSIVE, lr_a));
  transpose_launcher.region_requirements[0].add_field(FID_X);
  transpose_launcher.add_region_requirement(
      RegionRequirement(output_lp,  0/*projection ID*/,
        READ_WRITE, EXCLUSIVE, lr_b));
  transpose_launcher.region_requirements[1].add_field(FID_Y);

  FutureMap fm = runtime->execute_index_space(ctx, transpose_launcher);
  fm.wait_all_results();

  typedef std::pair<double, double> pairdd;
  double ts_start = DBL_MAX, ts_end = DBL_MIN;
  std::pair<double, double> times(ts_start, ts_end);

  for (GenericPointInRectIterator<1> pir(color_bounds); pir; pir++){
    std::pair<double, double> times(fm.get_result<pairdd>(DomainPoint::from_point<1>(pir.p)));
    ts_start = MIN(ts_start, times.first);
    ts_end = MAX(ts_end, times.second);
  }

  double max_time = ts_end - ts_start;
  double avg_time = max_time / iterations;

  double bytes = 2.0 * sizeof(double) * order * order;
  printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.0E-06 * bytes/avg_time, avg_time);

  /*********************************************************************
   ** Launch sub-task for results verification
   *********************************************************************/

  TaskLauncher check_launcher(TASKID_CHECK, TaskArgument(&args, sizeof(args)));
  check_launcher.add_region_requirement(
      RegionRequirement(lr_b, READ_ONLY, EXCLUSIVE, lr_b));
  check_launcher.region_requirements[0].add_field(FID_Y);

  runtime->execute_task(ctx, check_launcher);

  /*********************************************************************
  ** Do some clean-up
  *********************************************************************/

  runtime->destroy_logical_region(ctx, lr_a);
  runtime->destroy_logical_region(ctx, lr_b);
  runtime->destroy_field_space(ctx, ifs);
  runtime->destroy_field_space(ctx, ofs);
  runtime->destroy_index_space(ctx, is);
}

void init_a_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  // Check that the inputs look right since we have no
  // static checking to help us out.
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  RegionAccessor<AccessorType::Generic, double> accessor_a =
    regions[0].get_field_accessor(FID_X).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    accessor_a.write(DomainPoint::from_point<1>(pir.p), pir.p.x[0]);
  }
}

void init_b_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  // Check that the inputs look right since we have no
  // static checking to help us out.
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  RegionAccessor<AccessorType::Generic, double> accessor_b =
    regions[0].get_field_accessor(FID_Y).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    accessor_b.write(DomainPoint::from_point<1>(pir.p), 0.0);
  }
}

std::pair<double, double> transpose_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(BlockArgs));
  BlockArgs args = *static_cast<const BlockArgs *>(task->args);
  int n = args.order;
  int iterations = args.iterations;
  int blocks = args.tile_size;
  int num_partitions = args.num_partitions;
  int tiling = (blocks > 0) && (blocks < n);

  RegionAccessor<AccessorType::Generic, double> acc_a =
    regions[0].get_field_accessor(FID_X).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_b =
    regions[1].get_field_accessor(FID_Y).typeify<double>();

  Domain dom_a = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain dom_b = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());

  Rect<1> rect_a = dom_a.get_rect<1>();
  Rect<1> rect_b = dom_b.get_rect<1>();

  double ts_start = DBL_MAX; 
  double ts_end = DBL_MIN;
  double* a = 0;
  double* b = 0;

  {
    Rect<1> subrect_a; ByteOffset offsets_a[1];
    a = acc_a.raw_rect_ptr<1>(rect_a, subrect_a, offsets_a);
    assert(rect_a == subrect_a);
    Rect<1> subrect_b; ByteOffset offsets_b[1];
    b = acc_b.raw_rect_ptr<1>(rect_b, subrect_b, offsets_b);
    assert(rect_b == subrect_b);
  }

  int blockx = n / num_partitions;
  int MYTHREAD = task->index_point.point_data[0];
  int colstart = blockx * MYTHREAD;
  int myoffset_b = blockx * n * MYTHREAD;

  for (int iter = 0; iter<=iterations; iter++)
  {
    if (iter == 1)
      ts_start = wtime();
    int local_idx = 0;

    if(!tiling){
      for (int j = colstart; j < colstart+blockx; j++) {
        for (int i = 0; i < n; i++){
          b[local_idx++] += a[j+i*n];
          a[j+i*n] += 1.0;
        }
      }
    }
    else {
      for (int j = colstart; j < colstart+blockx; j+=blocks) {
        for (int i = 0; i < n; i+=blocks)
          for (int jj = j; jj < MIN(colstart+blockx, j+blocks); jj++)
            for (int ii = i; ii < MIN(n, i+blocks); ii++)
            {
              b[ii+jj*n-myoffset_b] += a[jj+ii*n];
              a[jj+ii*n] += 1.0;
            }
      }
    }
  }

  ts_end = wtime();

  return std::pair<double, double>(ts_start, ts_end);
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(BlockArgs));
  BlockArgs args = *static_cast<const BlockArgs *>(task->args);
  int n = args.order;
  int iterations = args.iterations;

  RegionAccessor<AccessorType::Generic, double> acc_b =
    regions[0].get_field_accessor(FID_Y).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  double abserr = 0.0;
  double epsilon=1.e-8; /* error tolerance */

  double addit = ((double)(iterations+1) * (double) (iterations))/2.0;

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    int index_b = pir.p.x[0];
    int i = index_b / n;
    int j = index_b % n;
    Point<1> p_b(index_b);
    double value = acc_b.read(DomainPoint::from_point<1>(p_b));
    abserr += ABS(value - (double)((j*n + i)*(iterations+1)+addit));
  }

#ifdef VERBOSE
  printf("Sum of absolute differences: %f\n",abserr);
#endif

  if (abserr < epsilon) {
    printf("Solution validates\n");
#ifdef VERBOSE
    printf("Squared errors: %f \n", abserr);
#endif
  }
  else {
    printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n",
        abserr, epsilon);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TASKID_TOPLEVEL);
  HighLevelRuntime::register_legion_task<task_toplevel>(TASKID_TOPLEVEL,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<init_a_task>(TASKID_INITIALIZE_A,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<init_b_task>(TASKID_INITIALIZE_B,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<std::pair<double,double>, transpose_task>(TASKID_TRANSPOSE,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<check_task>(TASKID_CHECK,
      Processor::LOC_PROC, true/*single*/, false/*index*/);

  HighLevelRuntime::start(argc, argv);
}
