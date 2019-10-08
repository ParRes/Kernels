#ifndef PRK_SYCL_HPP
#define PRK_SYCL_HPP

#include <cstdlib>
#include <iostream>

#include "CL/sycl.hpp"

namespace sycl = cl::sycl;

// prebuilt kernels are not required/not fully supported on hipSYCL and triSYCL
#if defined(TRISYCL) || defined(__HIPSYCL__)
#define PREBUILD_KERNEL 0
#else
#define PREBUILD_KERNEL 1
#endif

// not all SYCL implementations may support all device types.
// If an implementation does not find any devices based on a
// device selector, it will throw an exception.
// These macros can be used to check if there's any chance
// of an implementation targeting a CPU and GPU.
#if !defined(__HIPSYCL__) || defined(HIPSYCL_PLATFORM_CPU)
#define SYCL_TRY_CPU_QUEUE 1
#else
#define SYCL_TRY_CPU_QUEUE 0
#endif

// !defined(HIPSYCL_PLATFORM_CPU) = !( defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HCC) )
#if !defined(__HIPSYCL__) || !defined(HIPSYCL_PLATFORM_CPU)
#define SYCL_TRY_GPU_QUEUE 1
#else
#define SYCL_TRY_GPU_QUEUE 0
#endif

#if 0
#include "prk_opencl.h"
#define USE_OPENCL 1
#endif

namespace prk {

    // There seems to be an issue with the clang CUDA/HIP toolchains not having
    // std::abort() available
    void abort(void) {
#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HCC)
        abort();
#else
        std::abort();
#endif
    }

    namespace SYCL {

        void print_device_platform(const sycl::queue & q) {
#if !defined(TRISYCL) && !defined(__HIPSYCL__)
            auto device      = q.get_device();
            auto platform    = device.get_platform();
            std::cout << "SYCL Device:   " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "SYCL Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
#endif
        }

        bool has_spir(const sycl::queue & q) {
#if !defined(TRISYCL) && !defined(__HIPSYCL__)
            auto device = q.get_device();
            return device.has_extension(sycl::string_class("cl_khr_spir"));
#else
            return true;
#endif
        }

        bool has_ptx(const sycl::queue & q) {
#ifdef __COMPUTECPP__
            return true;
#else
            return false;
#endif
        }

        bool has_fp64(const sycl::queue & q) {
#if !defined(TRISYCL) && !defined(__HIPSYCL__)
            auto device      = q.get_device();
            return device.has_extension(sycl::string_class("cl_khr_fp64"));
#else
            return true;
#endif
        }

        void print_exception_details(sycl::exception & e) {
#ifdef __COMPUTECPP__
            std::cout << e.get_file_name() << std::endl;
            std::cout << e.get_line_number() << std::endl;
            std::cout << e.get_description() << std::endl;
            std::cout << e.get_cl_error_message() << std::endl;
            std::cout << e.get_cl_code() << std::endl;
#endif
        }

    } // namespace SYCL

} // namespace prk

#endif // PRK_SYCL_HPP
