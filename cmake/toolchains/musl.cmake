# Mirrors common/make.defs.musl's CC selection -- musl libc's gcc wrapper
# for static, non-glibc binaries. C only (the original make.defs.musl file
# doesn't set CXX or FC either -- musl-only PRK coverage is the C
# kernels). Not installed on this machine; provided for completeness.
#
#   CC=/opt/musl/1.1.16/gcc-7/bin/musl-gcc -std=c11 -static

set(CMAKE_C_COMPILER /opt/musl/1.1.16/gcc-7/bin/musl-gcc CACHE STRING "")
