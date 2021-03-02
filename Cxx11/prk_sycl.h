#ifndef PRK_SYCL_HPP
#define PRK_SYCL_HPP

#include <cstdlib>
#include <iostream>

#include "CL/sycl.hpp"

namespace sycl = cl::sycl;

//#ifdef __COMPUTECPP
//#include <SYCL/experimental.hpp>
//namespace syclx = cl::sycl::experimental;
//#endif

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
#if defined(TRISYCL) || defined(__HIPSYCL__) || defined(DPCPP_CUDA)
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

        // returns true if FP64 will not work
        bool print_gen12lp_helper(const sycl::queue & q) {
            auto d = q.get_device();
            auto s = d.get_info<sycl::info::device::name>();
            if ( s.find("Gen12LP") != std::string::npos) {
                bool e1=false;
                bool e2=false;
                auto c1 = std::getenv("IGC_EnableDPEmulation");
                auto c2 = std::getenv("OverrideDefaultFP64Settings");
                std::string s1{c1};
                std::string s2{c2};
                if (s1 != "1" || s2 != "1") {
                    std::cout << std::endl
                              << "You are using Gen12LP, which emulates FP64.\n"
                              << "Please try again with the following environment variables set:\n"
                              << "    export IGC_EnableDPEmulation=1\n"
                              << "    export OverrideDefaultFP64Settings=1\n"
                              << std::endl;
                    return true;
                }
            }
            return false;
        }

    } // namespace SYCL

} // namespace prk

#endif // PRK_SYCL_HPP
