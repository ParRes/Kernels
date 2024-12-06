Parallel STL support is not mature.  Currently, we support two implementations:

* Intel 18.0+
  - https://software.intel.com/en-us/articles/parallel-stl-parallel-algorithms-in-standard-template-library has details
  - https://github.com/intel/parallelstl.git is the open-source implementation that works with GCC and Clang (TBB is required).

* GCC 7.2+
  - std::execution is not supported.
  - one enables parallelism explicitly by switching from std to `__gnu_parallel namespace`.
  - one enables parallelism implicitly by using the `_GLIBCXX_PARALLEL` preprocessor symbol.
  - [GCC docs](https://gcc.gnu.org/onlinedocs/libstdc++/manual/parallel_mode_using.html) have details.

Future implementation targets may include:

* https://github.com/KhronosGroup/SyclParallelSTL
* http://thrust.github.io/

