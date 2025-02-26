#define RESTRICT __restrict__

static inline void transpose_block(double * B, const double * A, size_t block_order)
{
  for (size_t i=0; i<block_order; i++) {
    for (size_t j=0; j<block_order; j++) {
      B[i*block_order+j] += A[j*block_order+i];
    }
  }
} 

