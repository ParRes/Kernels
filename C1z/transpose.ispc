#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#define USE_TASKS 1

export void initialize(
    uniform const int order, uniform double A[], uniform double B[])
{
  foreach (i = 0 ... order, j = 0 ... order) {
    A[i * order + j] = (double)(i * order + j);
    B[i * order + j] = 0.0;
  }
}

task void transpose_task(
    uniform const int order, uniform double A[], uniform double B[])
{
  uniform const int i = taskIndex0;
  varying const int j = (taskIndex1 * programCount) + programIndex;

  if (j >= order)
    return;

  B[i * order + j] += A[j * order + i];
  A[j * order + i] += 1.0;
}

export void transpose(
    uniform const int order, uniform double A[], uniform double B[])
{
#if USE_TASKS
  uniform const int N = ceil(order / (uniform float)programCount);
  launch[order, N] transpose_task(order, A, B);
#else
  foreach (i = 0 ... order, j = 0 ... order) {
    B[i * order + j] += A[j * order + i];
    A[j * order + i] += 1.0;
  }
#endif
}

task void transpose_tiled_task(uniform const int order,
    uniform double A[],
    uniform double B[],
    uniform const int tile_size)
{
  uniform const int i_begin = taskIndex0 * tile_size;
  uniform const int j_begin = taskIndex1 * tile_size;

  uniform const int i_edge = i_begin + tile_size;
  uniform const int j_edge = j_begin + tile_size;

  uniform const int i_end = MIN(i_edge, order);
  uniform const int j_end = MIN(j_edge, order);

  foreach (i = i_begin... i_end, j = j_begin... j_end) {
    B[i * order + j] += A[j * order + i];
    A[j * order + i] += 1.0;
  }
}

export void transpose_tiled(uniform const int order,
    uniform double A[],
    uniform double B[],
    uniform const int tile_size)
{
#if USE_TASKS
  uniform const int N = ceil(order / (uniform float)tile_size);
  launch[N, N] transpose_tiled_task(order, A, B, tile_size);
#else
  foreach_tiled (i = 0 ... order, j = 0 ... order) {
    B[i * order + j] += A[j * order + i];
    A[j * order + i] += 1.0;
  }
#endif
}
