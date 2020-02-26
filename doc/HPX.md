#

```sh
cmake .. -DCMAKE_INSTALL_PREFIX=$PRK_DIR/Cxx11/hpx \
         -DCMAKE_CXX_COMPILER=/usr/local/Cellar/llvm/9.0.1/bin/clang++ \
         -DCMAKE_C_COMPILER=/usr/local/Cellar/llvm/9.0.1/bin/clang \
         -DHPX_WITH_TESTS:BOOL=Off \
         -DHPX_WITH_TESTS_BENCHMARKS:BOOL=Off \
         -DHPX_WITH_TESTS_EXAMPLES:BOOL=Off \
         -DHPX_WITH_TESTS_REGRESSIONS:BOOL=Off \
         -DHPX_WITH_TESTS_UNIT:BOOL=Off
make install
```
