#ifndef PRK_SYCL_HPP
#define PRK_SYCL_HPP

#include <cstdlib>
#include <iostream>

//#include <iterator> // std::distance
#include <boost/range/adaptor/indexed.hpp>

#include "CL/sycl.hpp"

#ifdef __COMPUTECPP__
#include "SYCL/experimental/usm.h"
#endif

#include "prk_util.h" // prk::vector

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
            auto device      = q.get_device();
            auto platform    = device.get_platform();
            std::cout << "SYCL Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
            std::cout << "SYCL Device:   " << device.get_info<sycl::info::device::name>() << std::endl;
#endif
        }

        bool has_fp64(const sycl::queue & q) {
#if defined(TRISYCL) || defined(__HIPSYCL__)
            return true;
#else
            auto device      = q.get_device();
            return device.has_extension(sycl::string_class("cl_khr_fp64"));
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

        class queues {

            private:
                std::vector<sycl::queue> list;

            public:
                queues(bool use_cpu = true, bool use_gpu = true)
                {
                    auto platforms = sycl::platform::get_platforms();
                    for (auto & p : platforms) {
                        auto pname = p.get_info<sycl::info::platform::name>();
                        std::cout << "*Platform: " << pname << std::endl;
                        if ( pname.find("Level-Zero") != std::string::npos) {
                            std::cout << "*Level Zero GPU skipped" << std::endl;
                            break;
                        }
                        if ( pname.find("Intel") == std::string::npos) {
                            std::cout << "*non-Intel skipped" << std::endl;
                            break;
                        }
                        auto devices = p.get_devices();
                        for (auto & d : devices ) {
                            std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
                            if ( d.is_cpu() && use_cpu ) {
                                std::cout << "**Device is CPU - adding to vector of queues" << std::endl;
                                list.push_back(sycl::queue(d));
                            }
                            if ( d.is_gpu() && use_gpu ) {
                                std::cout << "**Device is GPU - adding to vector of queues" << std::endl;
                                list.push_back(sycl::queue(d));
                            }
                        }
                    }
                }

                int size(void)
                {
                    return list.size();
                }

                void wait(int i)
                {
                    list[i].wait();
                }

                void waitall(void)
                {
                    for (auto & i : list) {
                        i.wait();
                    }
                }

                sycl::queue queue(int i) {
                    return this->list[i];
                }

                template <typename T>
                void allocate(std::vector<T*> & device_pointers,
                              size_t num_elements)
                {
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        device_pointers[i] = syclx::malloc_device<T>(num_elements, v);
                    }
                }

                template <typename T>
                void free(std::vector<T*> & device_pointers)
                {
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        syclx::free(device_pointers[i], v);
                    }
                }

                // UNUSED and UNTESTED
                template <typename T, typename B>
                void broadcast(std::vector<T*> & device_pointers,
                               const B & host_pointer,
                               size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = device_pointers[i];
                        auto source = &host_pointer[0];
                        v.memcpy(target, source, bytes);
                    }
                }

                // UNUSED and UNTESTED
                template <typename T, typename B, typename Op>
                void reduce(B & host_pointer,
                            const std::vector<T*> & device_pointers,
                            size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    auto temp = std::vector<T>(num_elements, 0);
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = device_pointers[i];
                        auto source = &host_pointer[0];
                        v.memcpy(temp, source, bytes);
                        for (size_t e=0; e<num_elements; ++e) {
                            target[e] = Op{}( target[e], temp[e] );
                        }
                    }
                }

                template <typename T, typename B>
                void gather(B & host_pointer,
                            const std::vector<T*> & device_pointers,
                            size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = &host_pointer[i * num_elements];
                        auto source = device_pointers[i];
                        v.memcpy(target, source, bytes);
                    }
                }

                template <typename T, typename B>
                void scatter(std::vector<T*> & device_pointers,
                             const B & host_pointer,
                             size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = device_pointers[i];
                        auto source = &host_pointer[i * num_elements];
                        v.memcpy(target, source, bytes);
                    }
                }

                // num_elements is defined the same as MPI
                // each device contributes np * num_elements
                // each device receives np * num_elements
                template <typename T, typename B>
                void alltoall(std::vector<T*> & device_pointers_out,
                              std::vector<T*> & device_pointers_in,
                              size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    // allocate np*np temp space on the host, because
                    // we cannot copy device-to-device if they are in
                    // different contexts.
                    // we can specialize for single-context later...
                    int np = this->list.size();
                    prk::vector<double> temp(num_elements * np * np);

                    // gather phase - contiguous
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = &temp[i * np * num_elements];
                        auto source = device_pointers_in[i];
                        v.memcpy(target, source, np * bytes);
                    }

                    // scatter phase - noncontiguous
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = device_pointers_out[i];
                        auto source = &temp[i * num_elements];
                        v.memcpy(target, source, bytes);
                    }

                }
        };

    } // namespace SYCL

} // namespace prk

#endif // PRK_SYCL_HPP
