#define RESTRICT __restrict__

static inline void transpose_block(double * RESTRICT B, const double * RESTRICT A, size_t block_order)
{
  for (size_t i=0; i<block_order; i++) {
    for (size_t j=0; j<block_order; j++) {
      B[i*block_order+j] += A[j*block_order+i];
    }
  }
} 

static inline void transpose_block(double * RESTRICT B, const double * RESTRICT A, size_t block_order, size_t tile_size)
{
  if (tile_size < block_order) {
    for (size_t it=0; it<block_order; it+=tile_size) {
      for (size_t jt=0; jt<block_order; jt+=tile_size) {
        for (size_t i=it; i<std::min(block_order,it+tile_size); i++) {
          for (size_t j=jt; j<std::min(block_order,jt+tile_size); j++) {
            B[i*block_order+j] += A[j*block_order+i];
          }
        }
      }
    }
  } else {
    transpose_block(B, A, block_order);
  }
}

#ifdef __NVCC__

const int tile_dim = 32;
const int block_rows = 8;

__global__ void transposeNoBankConflict(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    __shared__ double tile[tile_dim][tile_dim+1];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transposeCoalesced(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    __shared__ double tile[tile_dim][tile_dim];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transposeNaive(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[x*order + (y+j)] += A[(y+j)*order + x];
    }
}

__global__ void transposeSimple(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x < order) && (y < order)) {
        B[x*order + y] += A[y*order + x];
    }
}

__global__ void cuda_increment(const unsigned n, double * RESTRICT A)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] += 1.0;
    }
}

__device__ void transposeNoBankConflictDevice(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    __shared__ double tile[tile_dim][tile_dim+1];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transposeNoBankConflictBulk(int np, unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto r = blockIdx.z * blockDim.z + threadIdx.z;
    if (r < np) {
      const size_t offset = order * order * r;
      transposeNoBankConflictDevice(order, A + offset, B + offset);
      __syncthreads();
    }
}

__device__ void transposeCoalescedDevice(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    __shared__ double tile[tile_dim][tile_dim];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transposeCoalescedBulk(int np, unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto r = blockIdx.z * blockDim.z + threadIdx.z;
    if (r < np) {
      const size_t offset = order * order * r;
      transposeCoalescedDevice(order, A + offset, B + offset);
      __syncthreads();
    }
}

__device__ void transposeNaiveDevice(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[x*order + (y+j)] += A[(y+j)*order + x];
    }
}

__global__ void transposeNaiveBulk(int np, unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;
    auto r = blockIdx.z * blockDim.z + threadIdx.z;
    if (r < np) {
      const size_t offset = order * order * r;

      for (int j = 0; j < tile_dim; j+= block_rows) {
          B[offset + x*order + (y+j)] += A[offset + (y+j)*order + x];
      }
    }
}

__global__ void transposeSimpleBulk(int np, unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    //for (int r=0; r<np; r++) {
    auto r = blockIdx.z * blockDim.z + threadIdx.z;
    if (r < np) {
      const size_t offset = order * order * r;

      if ((x < order) && (y < order)) {
          B[offset + x*order + y] += A[offset + y*order + x];
      }
    }
}

#ifdef USE_NVSHMEM

__global__ void transpose_nvshmem_ptr(int variant, size_t block_size, int me, int np,
                                      unsigned block_order, const double * RESTRICT A, double * RESTRICT B)
{
    // block_size = block_order * block_order
    for (int r=0; r<np; r++) {
        const int recv_from = (me + r) % np;
        const size_t soffset = block_size * me;
        const size_t roffset = block_size * recv_from;
        const double * T = (double*)nvshmem_ptr(A + soffset, recv_from);
        if (variant==0) {
            transposeNaiveDevice(block_order, T, B + roffset);
        } else if (variant==1) {
            transposeCoalescedDevice(block_order, T, B + roffset);
        } else if (variant==2) {
            transposeNoBankConflictDevice(block_order, T, B + roffset);
        }
        __syncthreads();
    }
}

#endif // NVSHMEM

#endif // NVCC
