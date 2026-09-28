# common/make.defs.upcxx-hpx doesn't set CC/CXX/FC at all -- it only adds
# UPCXX/UPCXXFLAG and HPXCXX/HPXFLAG on top of whatever base toolchain
# (e.g. make.defs.gcc) is otherwise in use, for the small number of UPC++/
# HPX-specific targets. Neither is wired into any of this project's CMake
# ports yet (no PRK_ENABLE_UPCXX/HPX target group exists), so there's
# nothing for a toolchain file to select here -- this file intentionally
# sets no compiler and exists only for parity with the other toolchain
# files/make.defs.* variants. Combine with e.g. gcc.cmake for the base
# compiler once/if UPC++ or HPX targets are added.
