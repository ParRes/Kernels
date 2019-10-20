# RAJA README

## IBM POWER9 + NVIDIA V100

```
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/RAJA/install-cuda \
         -DCMAKE_CXX_COMPILER=xlc++_r -DCMAKE_C_COMPILER=xlc_r \
         -DENABLE_OPENMP=On -DENABLE_TARGET_OPENMP=On -DOpenMP_CXX_FLAGS="-qsmp -qoffload" \
         -DENABLE_CUDA=On -DCUDA_ARCH=sm_70
 && make -j install
```

Optional extras: `-qsuppress=1500-030` or `-qmaxmem=-1`

