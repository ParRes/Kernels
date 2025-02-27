#ifndef PRK_NVSHMEM_HPP
#define PRK_NVSHMEM_hpp

#include <cstdio>
#include <cstdlib>
#include <cinttypes>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <limits>
#include <type_traits>
#include <utility>

#include <nvshmem.h>
#include <nvshmemx.h>

namespace prk {
    namespace NVSHMEM {

        [[noreturn]] void abort(int errorcode = -1) {
            nvshmem_global_exit(errorcode);
            std::abort(); // unreachable
        }

        class state {
          public:
            state(int * argc = NULL, char*** argv = NULL) {
                nvshmem_init();
            }

            ~state(void) {
                nvshmem_finalize();
            }
        };

        int rank(void) {
            return nvshmem_my_pe();
        }

        int size(void) {
            return nvshmem_n_pes();
        }

        void barrier(void) {
            nvshmem_barrier_all();
        }

        template <typename T>
        void broadcast(T * buffer, int root = 0, size_t count = 1) {
            if (nvshmem_broadcastmem(NVSHMEM_TEAM_WORLD, buffer, buffer, count * sizeof(T), root)) {
                throw std::runtime_error("nvshmem_broadcastmem failed");
            }
        }

        template <typename T>
        void alltoall(T * rbuffer,const T * sbuffer,  size_t count = 1) {
            if (nvshmem_alltoallmem(NVSHMEM_TEAM_WORLD, rbuffer, sbuffer, count * sizeof(T))) {
                throw std::runtime_error("nvshmem_alltoallmem failed");
            }
        }

        template <typename T>
        void put(T * dest, const T * source, size_t count, int pe) {
            nvshmem_putmem(dest, source, count * sizeof(T), pe);
        }

        template <typename T>
        void get(T * dest, const T * source, size_t count, int pe) {
            nvshmem_getmem(dest, source, count * sizeof(T), pe);
        }

        template <typename T>
        T * allocate(size_t count) {
            T * ptr = (T*)nvshmem_malloc(count * sizeof(T));
            if (!ptr) {
                throw std::runtime_error("nvshmem_malloc failed");
            }
            return ptr;
        }

        template <typename T>
        void free(T * ptr) {
            nvshmem_free(ptr);
        }

        double min(double in) {
            double out;
            nvshmem_double_min_reduce(NVSHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        int min(int in) {
            int out;
            nvshmem_int_min_reduce(NVSHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        double max(double in) {
            double out;
            nvshmem_double_max_reduce(NVSHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        int max(int in) {
            int out;
            nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        double sum(double in) {
            double out;
            nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        double avg(double in) {
            double out = sum(in);
            out /= prk::NVSHMEM::size();
            return out;
        }

        void stats(double in, double * min, double * max, double * avg) {
            *min = prk::NVSHMEM::min(in);
            *max = prk::NVSHMEM::max(in);
            *avg = prk::NVSHMEM::avg(in);
        }

        bool is_same(int in) {
            int min = std::numeric_limits<int>::max();
            int max = std::numeric_limits<int>::min();
            min = prk::NVSHMEM::min(in);
            max = prk::NVSHMEM::max(in);
            return (min == max);
        }

        bool is_same(double in) {
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::min();
            min = prk::NVSHMEM::min(in);
            max = prk::NVSHMEM::max(in);
            return (min == max);
        }

        template <typename TF, typename TI>
        void print_matrix(const TF * matrix, TI rows, TI cols, const std::string label = "") {
            int me = prk::NVSHMEM::rank();
            int np = prk::NVSHMEM::size();

            prk::NVSHMEM::barrier();

            for (int r = 0; r < np; ++r) {
                if (me == r) {
                    std::cerr << label << "\n";
                    for (TI i = 0; i < rows; ++i) {
                        for (TI j = 0; j < cols; ++j) {
                            std::cerr << matrix[i * cols + j] << " ";
                        }
                        std::cerr << "\n";
                    }
                    std::cerr << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                prk::NVSHMEM::barrier();
            }
        }

    } // NVSHMEM namespace
} // prk namespace

#endif // PRK_NVSHMEM_HPP 
