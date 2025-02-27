#ifndef PRK_OSHMEM_HPP
#define PRK_OSHMEM_HPP

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

#include <shmem.h>

namespace prk {
    namespace SHMEM {

        [[noreturn]] void abort(int errorcode = -1) {
            shmem_global_exit(errorcode);
            std::abort(); // unreachable
        }

        class state {
          public:
            state(int * argc = NULL, char*** argv = NULL) {
                shmem_init();
            }

            ~state(void) {
                shmem_finalize();
            }
        };

        int rank(void) {
            return shmem_my_pe();
        }

        int size(void) {
            return shmem_n_pes();
        }

        void barrier(void) {
            shmem_barrier_all();
        }

        template <typename T>
        void broadcast(T * buffer, int root = 0, size_t count = 1) {
            if (shmem_broadcastmem(SHMEM_TEAM_WORLD, buffer, buffer, count * sizeof(T), root)) {
                throw std::runtime_error("shmem_broadcastmem failed");
            }
        }

        template <typename T>
        void alltoall(T * rbuffer,const T * sbuffer,  size_t count = 1) {
            if (shmem_alltoallmem(SHMEM_TEAM_WORLD, rbuffer, sbuffer, count * sizeof(T))) {
                throw std::runtime_error("shmem_alltoallmem failed");
            }
        }

        template <typename T>
        void put(T * dest, const T * source, size_t count, int pe) {
            shmem_putmem(dest, source, count * sizeof(T), pe);
        }

        template <typename T>
        void get(T * dest, const T * source, size_t count, int pe) {
            shmem_getmem(dest, source, count * sizeof(T), pe);
        }

        template <typename T>
        T * allocate(size_t count) {
            T * ptr = (T*)shmem_malloc(count * sizeof(T));
            if (!ptr) {
                throw std::runtime_error("shmem_malloc failed");
            }
            return ptr;
        }

        template <typename T>
        void free(T * ptr) {
            shmem_free(ptr);
        }

        double min(double in) {
            double out;
            shmem_double_min_reduce(SHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        int min(int in) {
            int out;
            shmem_int_min_reduce(SHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        double max(double in) {
            double out;
            shmem_double_max_reduce(SHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        int max(int in) {
            int out;
            shmem_int_max_reduce(SHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        double sum(double in) {
            double out;
            shmem_double_sum_reduce(SHMEM_TEAM_WORLD, &out, &in, 1);
            return out;
        }

        double avg(double in) {
            double out = sum(in);
            out /= prk::SHMEM::size();
            return out;
        }

        void stats(double in, double * min, double * max, double * avg) {
            *min = prk::SHMEM::min(in);
            *max = prk::SHMEM::max(in);
            *avg = prk::SHMEM::avg(in);
        }

        bool is_same(int in) {
            int min = std::numeric_limits<int>::max();
            int max = std::numeric_limits<int>::min();
            min = prk::SHMEM::min(in);
            max = prk::SHMEM::max(in);
            return (min == max);
        }

        bool is_same(double in) {
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::min();
            min = prk::SHMEM::min(in);
            max = prk::SHMEM::max(in);
            return (min == max);
        }

        template <typename TF, typename TI>
        void print_matrix(const TF * matrix, TI rows, TI cols, const std::string label = "") {
            int me = prk::SHMEM::rank();
            int np = prk::SHMEM::size();

            prk::SHMEM::barrier();

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
                prk::SHMEM::barrier();
            }
        }

    } // SHMEM namespace
} // prk namespace

#endif // PRK_OSHMEM_HPP 