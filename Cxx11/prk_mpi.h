#ifndef PRK_MPI_HPP
#define PRK_MPI_HPP

#include <cstdio>
#include <cstdlib>
#include <cinttypes>

#include <iostream>
#include <vector>
#include <string>

#include <type_traits>

#include <mpi.h>

#define ENABLE_SHM 1
#define STL_VECTOR_API 1

namespace prk
{
    namespace MPI
    {
        double wtime(void) { return MPI_Wtime(); }
        double wtick(void) { return MPI_Wtick(); }

        void abort(int errorcode = -1, MPI_Comm comm = MPI_COMM_WORLD)
        {
            MPI_Abort(comm, errorcode);
            std::abort(); // unreachable
        }

        void check(int errorcode)
        {
            if (errorcode==MPI_SUCCESS) {
                return;
            } else {
                int resultlen;

                char errorcode_string[MPI_MAX_ERROR_STRING];
                char errorclass_string[MPI_MAX_ERROR_STRING];

                int errorclass;
                MPI_Error_class(errorcode, &errorclass);

                MPI_Error_string(errorclass, errorclass_string, &resultlen);
                std::cerr << "MPI error: class " << errorclass << ", " << errorclass_string << std::endl;

                MPI_Error_string(errorcode, errorcode_string, &resultlen);
                std::cerr << "MPI error: code " << errorcode << ", " << errorcode_string << std::endl;

                prk::MPI::abort(errorcode);
            }
        }

        class state {

          private:
            MPI_Comm node_comm_;

          public:
            state(int * argc = NULL, char*** argv = NULL) {
                int is_init, is_final;
                MPI_Initialized(&is_init);
                MPI_Finalized(&is_final);
                if (!is_init && !is_final) {
                    if (argv==NULL && argc!=NULL) {
                        std::cerr << "argv is NULL but argc is not!" << std::endl;
                        std::abort();
                    }
                    MPI_Init(argc,argv);
                    prk::MPI::check( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &this->node_comm_) );
                }
            }

            ~state(void) {
                int is_init, is_final;
                MPI_Initialized(&is_init);
                MPI_Finalized(&is_final);
                if (is_init && !is_final) {
                    prk::MPI::check( MPI_Comm_free(&this->node_comm_) );
                    MPI_Finalize();
                }
            }

            MPI_Comm node_comm(void) {
                // this is a handle so we can always return a copy of the private instance
                return this->node_comm_;
            }
        };

        int rank(MPI_Comm comm = MPI_COMM_WORLD) {
            int rank;
            prk::MPI::check( MPI_Comm_rank(comm,&rank) );
            return rank;
        }

        int size(MPI_Comm comm = MPI_COMM_WORLD) {
            int size;
            prk::MPI::check( MPI_Comm_size(comm,&size) );
            return size;
        }

        void barrier(MPI_Comm comm = MPI_COMM_WORLD) {
            prk::MPI::check( MPI_Barrier(comm) );
        }

        double min(double in, MPI_Comm comm = MPI_COMM_WORLD) {
            double out;
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_MIN, comm) );
            return out;
        }

        double max(double in, MPI_Comm comm = MPI_COMM_WORLD) {
            double out;
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_MAX, comm) );
            return out;
        }

        double sum(double in, MPI_Comm comm = MPI_COMM_WORLD) {
            double out;
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, comm) );
            return out;
        }

        double avg(double in, MPI_Comm comm = MPI_COMM_WORLD) {
            double out;
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, comm) );
            out /= prk::MPI::size(comm);
            return out;
        }

        void stats(double in, double * min, double * max, double * avg, MPI_Comm comm = MPI_COMM_WORLD) {
            prk::MPI::check( MPI_Allreduce(&in, min, 1, MPI_DOUBLE, MPI_MIN, comm) );
            prk::MPI::check( MPI_Allreduce(&in, max, 1, MPI_DOUBLE, MPI_MAX, comm) );
            prk::MPI::check( MPI_Allreduce(&in, avg, 1, MPI_DOUBLE, MPI_SUM, comm) );
            *avg /= prk::MPI::size(comm);
        }

        bool is_same(int in, MPI_Comm comm = MPI_COMM_WORLD) {
            int min=INT_MAX, max=0;
            prk::MPI::check( MPI_Allreduce(&in, &min, 1, MPI_INT, MPI_MIN, comm) );
            prk::MPI::check( MPI_Allreduce(&in, &max, 1, MPI_INT, MPI_MAX, comm) );
            return (min==max);
        }

        bool is_same(size_t in, MPI_Comm comm = MPI_COMM_WORLD) {
            size_t min=SIZE_MAX, max=0;
            MPI_Datatype dt = (std::is_signed<size_t>() ? MPI_INT64_T : MPI_UINT64_T);
            prk::MPI::check( MPI_Allreduce(&in, &min, 1, dt, MPI_MIN, comm) );
            prk::MPI::check( MPI_Allreduce(&in, &max, 1, dt, MPI_MAX, comm) );
            return (min==max);
        }

        size_t sum(size_t in, MPI_Comm comm = MPI_COMM_WORLD) {
            size_t out;
            MPI_Datatype dt = (std::is_signed<size_t>() ? MPI_INT64_T : MPI_UINT64_T);
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, dt, MPI_SUM, comm) );
            return out;
        }

        template <typename T>
        class vector {

          private:
              size_t global_size_;
              size_t local_size_;

#if ENABLE_SHM
              MPI_Comm node_comm_;
              MPI_Win shm_win_;
#endif
              MPI_Comm comm_;
              MPI_Win distributed_win_;

              T * local_pointer_;

          public:
            vector(size_t global_size, T fill_value = 0, MPI_Comm comm = MPI_COMM_WORLD)
            {
                prk::MPI::check( MPI_Comm_dup(comm, &comm_) );

                int np = prk::MPI::size(comm_);
                int me = prk::MPI::rank(comm_);

                bool consistency = prk::MPI::is_same(global_size, comm_);
                if (!consistency) {
                    if (me == 0) std::cerr << "global size inconsistent!\n"
                                           << " rank = " << me << ", global size = " << global_size << std::endl;
                    prk::MPI::abort();
                }

                global_size_ = global_size;
                local_size_ = global_size_ / np;
                const size_t remainder  = global_size_ % np;
                if (me < remainder) local_size_++;

                const size_t verify_global_size = sum(local_size_, comm_);
                if (global_size != verify_global_size) {
                    if (me == 0) std::cerr << "global size inconsistent!\n"
                                           << " expected: " << global_size << "\n"
                                           << " actual:   " << verify_global_size << "\n";
                    std::cerr << "rank = " << me << ", local size = " << local_size_ << std::endl;
                    prk::MPI::abort();
                }

                size_t local_bytes = local_size_ * sizeof(T);
#if ENABLE_SHM
                prk::MPI::check( MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm_) );
                prk::MPI::check( MPI_Win_allocate_shared(local_bytes, 1, MPI_INFO_NULL, node_comm_,
                                                         &local_pointer_, &shm_win_) );

                prk::MPI::check( MPI_Win_create(local_pointer_, local_bytes, 1, MPI_INFO_NULL, comm_,
                                                &distributed_win_) );
#else
                prk::MPI::check( MPI_Win_allocate(local_bytes, 1, MPI_INFO_NULL, comm_,
                                                  &local_pointer_, &distributed_win_) );
#endif

                for (size_t i=0; i < local_size_; ++i) {
                    local_pointer_[i] = fill_value;
                }
            }

            ~vector(void)
            {
                prk::MPI::check( MPI_Win_free(&distributed_win_) );
#if ENABLE_SHM
                prk::MPI::check( MPI_Win_free(&shm_win_) );
                prk::MPI::check( MPI_Comm_free(&node_comm_) );
#endif
                prk::MPI::check( MPI_Comm_free(&comm_) );
            }

            T * local_pointer(size_t offset = 0)
            {
                return &local_pointer_[offset];
            }

            size_t local_size(void)
            {
                return local_size_;
            }

#if STL_VECTOR_API
            size_t size(void)
            {
                return local_size_;
            }

            T& operator[](size_t offset)
            {
                return local_pointer_[offset];
            }
#endif
        };

    } // MPI namespace

} // prk namespace

#endif // PRK_MPI_HPP
