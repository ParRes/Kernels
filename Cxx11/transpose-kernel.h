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

__global__ void transposeSimple2(unsigned order, const double * RESTRICT A, double * RESTRICT B)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < order; i += blockDim.x * gridDim.x) {
      for (unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; j < order; j += blockDim.y * gridDim.y) {
        B[i*order + j] += A[j*order + i];
      }
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

__device__ void transposeNaiveDevice_get(unsigned order, /* double * RESTRICT T, */ const double * RESTRICT A, double * RESTRICT B, int recv_from)
{
    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    //T = (double*)nvshmem_ptr(A, recv_from); // this works but is not the goal

    for (int j = 0; j < tile_dim; j+= block_rows) {
        //nvshmem_getmem(&T[(y+j)*order + x], &A[(y+j)*order + x], sizeof(double), recv_from); // this works but why use a big buffer?
        //B[x*order + (y+j)] += T[(y+j)*order + x];
        double T;
        nvshmem_getmem(&T, &A[(y+j)*order + x], sizeof(double), recv_from);
        B[x*order + (y+j)] += T;
    }
}

__device__ void transposeCoalescedDevice_get(unsigned order, const double * RESTRICT A, double * RESTRICT B, int recv_from)
{
    __shared__ double tile[tile_dim][tile_dim];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       double T;
       nvshmem_getmem(&T, &A[(y+j)*order + x], sizeof(double), recv_from);
       tile[threadIdx.y+j][threadIdx.x] = T;
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__device__ void transposeNoBankConflictDevice_get(unsigned order, const double * RESTRICT A, double * RESTRICT B, int recv_from)
{
    __shared__ double tile[tile_dim][tile_dim+1];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       double T;
       nvshmem_getmem(&T, &A[(y+j)*order + x], sizeof(double), recv_from);
       tile[threadIdx.y+j][threadIdx.x] = T;
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transpose_nvshmem_get(int variant, size_t block_size, int me, int np,
                                      unsigned block_order, const double * RESTRICT A, double * RESTRICT B, double * RESTRICT T)
{
    // block_size = block_order * block_order
    for (int r=0; r<np; r++) {
        const int recv_from = (me + r) % np;
        const size_t soffset = block_size * me;
        const size_t roffset = block_size * recv_from;
        //const double * T = (double*)nvshmem_ptr(A + soffset, recv_from);
        //nvshmemx_getmem(T, A + soffset, block_order * block_order * sizeof(double), recv_from);

        if (variant==0) {
            transposeNaiveDevice_get(block_order, /* T, */ A + soffset, B + roffset, recv_from);
        } else if (variant==1) {
            transposeCoalescedDevice_get(block_order, A + soffset, B + roffset, recv_from);
        } else if (variant==2) {
            transposeNoBankConflictDevice_get(block_order, A + soffset, B + roffset, recv_from);
        }
        __syncthreads();
    }
}

// PUT-based transpose kernels for NVSHMEM
__device__ void transposeNaiveDevice_put(unsigned order, const double * RESTRICT A, double * RESTRICT B, int send_to)
{
    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        // Transpose and put directly to remote PE
        double T = A[(y+j)*order + x];
        nvshmem_putmem(&B[x*order + (y+j)], &T, sizeof(double), send_to);
    }
}

__device__ void transposeCoalescedDevice_put(unsigned order, const double * RESTRICT A, double * RESTRICT B, int send_to)
{
    __shared__ double tile[tile_dim][tile_dim];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    // Load data into shared memory
    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
    }

    __syncthreads();

    // Transpose indices for output
    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    // Write transposed data to remote PE
    for (int j = 0; j < tile_dim; j+= block_rows) {
        double T = tile[threadIdx.x][threadIdx.y + j];
        nvshmem_putmem(&B[(y+j)*order + x], &T, sizeof(double), send_to);
    }
}

__device__ void transposeNoBankConflictDevice_put(unsigned order, const double * RESTRICT A, double * RESTRICT B, int send_to)
{
    __shared__ double tile[tile_dim][tile_dim+1];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    // Load data into shared memory
    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
    }

    __syncthreads();

    // Transpose indices for output
    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    // Write transposed data to remote PE
    for (int j = 0; j < tile_dim; j+= block_rows) {
        double T = tile[threadIdx.x][threadIdx.y + j];
        nvshmem_putmem(&B[(y+j)*order + x], &T, sizeof(double), send_to);
    }
}

__global__ void transpose_nvshmem_put(int variant, size_t block_size, int me, int np,
                                      unsigned block_order, const double * RESTRICT A, double * RESTRICT B)
{
    // Staged communication approach inspired by SHMEM implementation
    // Phase 0: local transpose (done on diagonal block)
    if (blockIdx.z == 0) {
        const size_t offset = block_size * me;
        if (variant==0) {
            transposeNaiveDevice(block_order, A + offset, B + offset);
        } else if (variant==1) {
            transposeCoalescedDevice(block_order, A + offset, B + offset);
        } else if (variant==2) {
            transposeNoBankConflictDevice(block_order, A + offset, B + offset);
        }
    }
    
    __syncthreads();
    
    // Phases 1 to np-1: remote communication using PUT
    for (int phase = 1; phase < np; phase++) {
        const int send_to = (me - phase + np) % np;
        const int recv_from = (me + phase) % np;
        
        const size_t soffset = block_size * send_to;    // source offset in A
        const size_t roffset = block_size * recv_from;  // destination offset in B
        
        if (variant==0) {
            transposeNaiveDevice_put(block_order, A + soffset, B + roffset, recv_from);
        } else if (variant==1) {
            transposeCoalescedDevice_put(block_order, A + soffset, B + roffset, recv_from);
        } else if (variant==2) {
            transposeNoBankConflictDevice_put(block_order, A + soffset, B + roffset, recv_from);
        }
        
        // Synchronize between phases
        __syncthreads();
    }
}

#endif // NVSHMEM

#endif // NVCC
