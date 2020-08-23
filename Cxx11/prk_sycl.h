#ifndef PRK_SYCL_HPP
#define PRK_SYCL_HPP

#include <cstdlib>
#include <iostream>

#include "CL/sycl.hpp"

#ifdef __COMPUTECPP__
#include "SYCL/experimental/usm.h"
#endif

namespace sycl = cl::sycl;

#ifdef __COMPUTECPP__
namespace syclx = cl::sycl::experimental;
#else
namespace syclx = cl::sycl;
#endif

#ifdef PRK_SYCL_USE_FLOAT
typedef float prk_float;
#else
typedef double prk_float;
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define OPENCL_CONSTANT __attribute__((opencl_constant))
#else
#define OPENCL_CONSTANT
#endif

// EXAMPLE OF PRINTF DEBUGGING IN SYCL DEVICE CODE
//static const OPENCL_CONSTANT char format[] = "%d:%lf,%lf,%lf\n";
//sycl::intel::experimental::printf(format, g, p_A[i], p_B[i], p_C[i]);

// prebuilt kernels are not required/not fully supported on hipSYCL and triSYCL
#if defined(TRISYCL) || defined(__HIPSYCL__) || defined(DPCPP)
#define PREBUILD_KERNEL 0
#else
#define PREBUILD_KERNEL 1
#endif

namespace prk {

    // There seems to be an issue with the clang CUDA/HIP toolchains not having
    // std::abort() available
    void Abort(void) {
#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HCC)
        abort();
#else
        std::abort();
#endif
    }

    namespace SYCL {

        void print_device_platform(const sycl::queue & q) {
#if ! ( defined(TRISYCL) || defined(__HIPSYCL__) )
            auto d = q.get_device();
            auto p = d.get_platform();
            std::cout << "SYCL Device:   " << d.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "SYCL Platform: " << p.get_info<sycl::info::platform::name>() << std::endl;
#endif
        }

        bool has_fp64(const sycl::queue & q) {
#if defined(TRISYCL) || defined(__HIPSYCL__)
            return true;
#else
            auto device = q.get_device();
            return device.has_extension(sycl::string_class("cl_khr_fp64"));
#endif
        }

        void print_exception_details(sycl::exception & e) {
            std::cout << e.what() << std::endl;
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
