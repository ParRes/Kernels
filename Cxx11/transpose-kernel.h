#define RESTRICT __restrict__

static inline void transpose_block(double * B, const double * A, size_t block_order)
{
  for (size_t i=0; i<block_order; i++) {
    for (size_t j=0; j<block_order; j++) {
      B[i*block_order+j] += A[j*block_order+i];
    }
  }
} 

static inline void transpose_block(double * B, const double * A, size_t block_order, size_t tile_size)
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
