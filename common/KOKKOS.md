# Kokkos README

## IBM POWER9 + NVIDIA V100

If you do not enable GPU arch >5, it fails at runtime.

If you do not enable lambda support, `parallel_reduce` will not compile.

```
cmake .. -DKokkos_ENABLE_CUDA=True \
         -DCMAKE_CXX_COMPILER=$HOME/KOKKOS/git/bin/nvcc_wrapper \
         -DCMAKE_INSTALL_PREFIX=$HOME/KOKKOS/install-cuda \
         -DKokkos_ARCH_POWER9=ON \
         -DKokkos_ARCH_VOLTA70=ON \
         -DKokkos_ENABLE_CUDA_LAMBDA=ON \
 && make -j install
```
