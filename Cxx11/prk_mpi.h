#ifndef PRK_MPI_HPP
#define PRK_MPI_HPP

#include <cstdio>
#include <cstdlib>
#include <cinttypes>

#include <iostream>
#include <string>
#include <vector>
#include <numeric> // exclusive_scan
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

        [[noreturn]] void abort(int errorcode = -1, MPI_Comm comm = MPI_COMM_WORLD)
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
                        prk::MPI::abort();
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

              int np_, me_; // global size and rank

              T * local_pointer_;

              MPI_Datatype dt_;

              size_t my_global_offset_begin_;
              size_t my_global_offset_end_;

              std::vector<size_t> global_offsets_;

          public:
            vector(size_t global_size, T fill_value = 0, MPI_Comm comm = MPI_COMM_WORLD)
            {
                prk::MPI::check( MPI_Comm_dup(comm, &comm_) );

                np_ = prk::MPI::size(comm_);
                me_ = prk::MPI::rank(comm_);

                bool consistency = prk::MPI::is_same(global_size, comm_);
                if (!consistency) {
                    if (me_ == 0) std::cerr << "global size inconsistent!\n"
                                           << " rank = " << me_ << ", global size = " << global_size << std::endl;
                    prk::MPI::abort();
                }

                global_size_ = global_size;
                local_size_ = global_size_ / np_;
                const size_t remainder  = global_size_ % np_;
                if (me_ < remainder) local_size_++;

                {
                    MPI_Datatype dt = (std::is_signed<size_t>() ? MPI_INT64_T : MPI_UINT64_T);
                    std::vector<size_t> global_sizes(np_);   // in
                    global_offsets_.resize(np_);             // out
                    prk::MPI::check( MPI_Allgather(&local_size_, 1, dt, global_sizes.data(), 1, dt, comm_) );
                    std::exclusive_scan( global_sizes.cbegin(), global_sizes.cend(), global_offsets_.begin(), 0);
                }
                my_global_offset_begin_ = global_offsets_[me_];
                my_global_offset_end_   = (me_ != np_-1) ? global_offsets_[me_+1] : global_size_;
#if 0
                if (me_ == 0) {
                    std::cout << "global offsets = ";
                    for ( size_t i=0; i<global_offsets_.size(); ++i) {
                        std::cout << global_offsets_[i] << ",";
                    }
                    std::cout << std::endl;
                }
                std::cout << "global offsets {begin,end} = " << my_global_offset_begin_ << "," << my_global_offset_end_ << std::endl;
#endif

                const size_t verify_global_size = sum(local_size_, comm_);
                if (global_size != verify_global_size) {
                    if (me_ == 0) std::cerr << "global size inconsistent!\n"
                                           << " expected: " << global_size << "\n"
                                           << " actual:   " << verify_global_size << "\n";
                    std::cerr << "rank = " << me_ << ", local size = " << local_size_ << std::endl;
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
                prk::MPI::check( MPI_Win_lock_all(MPI_MODE_NOCHECK, distributed_win_) );

                for (size_t i=0; i < local_size_; ++i) {
                    local_pointer_[i] = fill_value;
                }

                // built-in datatypes are not compile-time constants...
                if ( std::is_same<T,double>::value ) {
                    dt_ = MPI_DOUBLE;
                } else if ( std::is_same<T,float>::value ) {
                    dt_ = MPI_FLOAT;
                } else {
                    dt_ = MPI_DATATYPE_NULL;
                    std::cerr << "unknown type" << std::endl;
                    prk::MPI::abort();
                }
            }

            ~vector(void) noexcept
            {
                prk::MPI::check( MPI_Win_unlock_all(distributed_win_) );
                prk::MPI::check( MPI_Win_free(&distributed_win_) );
#if ENABLE_SHM
                prk::MPI::check( MPI_Win_free(&shm_win_) );
                prk::MPI::check( MPI_Comm_free(&node_comm_) );
#endif
                prk::MPI::check( MPI_Comm_free(&comm_) );
            }

            T * local_pointer(size_t offset = 0) noexcept
            {
                return &local_pointer_[offset];
            }

            size_t local_size(void) noexcept
            {
                return local_size_;
            }

#if STL_VECTOR_API
            size_t size(void) noexcept
            {
                return local_size_;
            }

            T& operator[](size_t local_offset) noexcept
            {
                return local_pointer_[local_offset];
            }

            constexpr T * data(void) noexcept
            {
                return local_pointer_;
            }
#endif
            // read-only element-wise access to remote data
            T const get(size_t global_offset)
            {
                for (size_t i=0; i < np_; ++i) {
                    if (global_offsets_[i] <= global_offset &&
                            ( (i+1)<np_ ? global_offset < global_offsets_[(i+1)] : global_offset < global_size_)
                        ) {
                        //std::cout << "global_offset " << global_offset << " found at rank " << i << "\n";
                        T data;
                        MPI_Request req;
                        MPI_Aint win_offset = global_offset - global_offsets_[i];
                        prk::MPI::check( MPI_Rget(&data, 1, dt_, i /* rank */, win_offset * sizeof(T), 1, dt_, distributed_win_, &req) );
                        prk::MPI::check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
                        return data;
                    }
                }
                std::cerr << "global_offset " << global_offset << " not found!" << std::endl;
                prk::MPI::abort();
            }

            // element-wise write access to remote data
            // non-temporal i.e. no remote visibility unless fenced
            void put(size_t global_offset, T data)
            {
                for (size_t i=0; i < np_; ++i) {
                    if (global_offsets_[i] <= global_offset &&
                            ( (i+1)<np_ ? global_offset < global_offsets_[(i+1)] : global_offset < global_size_)
                        ) {
                        //std::cout << "global_offset " << global_offset << " found at rank " << i << "\n";
                        MPI_Request req;
                        MPI_Aint win_offset = global_offset - global_offsets_[i];
                        prk::MPI::check( MPI_Rput(&data, 1, dt_, i /* rank */, win_offset * sizeof(T), 1, dt_, distributed_win_, &req) );
                        prk::MPI::check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
                        return;
                    }
                }
                std::cerr << "global_offset " << global_offset << " not found!" << std::endl;
                prk::MPI::abort();
            }

            // element-wise write access to remote data
            // non-temporal i.e. no remote visibility unless fenced
            void add(size_t global_offset, T data)
            {
                for (size_t i=0; i < np_; ++i) {
                    if (global_offsets_[i] <= global_offset &&
                            ( (i+1)<np_ ? global_offset < global_offsets_[(i+1)] : global_offset < global_size_)
                        ) {
                        //std::cout << "global_offset " << global_offset << " found at rank " << i << "\n";
                        MPI_Request req;
                        MPI_Aint win_offset = global_offset - global_offsets_[i];
                        prk::MPI::check( MPI_Raccumulate(&data, 1, dt_, i /* rank */, win_offset * sizeof(T), 1, dt_, MPI_SUM, distributed_win_, &req) );
                        prk::MPI::check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
                        return;
                    }
                }
                std::cerr << "global_offset " << global_offset << " not found!" << std::endl;
                prk::MPI::abort();
            }

            // synchronize all outstanding writes
            void fence(void)
            {
                prk::MPI::check( MPI_Win_flush_all(distributed_win_) );
            }

            // read-only access to any data in the vector, include remote data
            T const operator()(size_t global_offset) noexcept
            {
                // if the data is in local memory, return a reference immediately
                if (my_global_offset_begin_ <= global_offset && global_offset < my_global_offset_end_) {
                    return local_pointer_[global_offset];
                }
                /* TODO: add intranode, interprocess path here
                else if { } */
                // remote memory fetch
                else {
                    return get(global_offset);
                }
            }

        };

    } // MPI namespace

} // prk namespace

#endif // PRK_MPI_HPP
