#ifndef PRK_MPI_HPP
#define PRK_MPI_HPP

#include <cstdio>
#include <cstdlib>
#include <cinttypes>

#include <iostream>
#include <string>
#include <vector>
#include <numeric> // exclusive_scan
#include <limits>
#include <type_traits>
#include <utility>

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

        template <typename T>
        MPI_Datatype get_MPI_Datatype(T t) { 
            std::cerr << "get_MPI_Datatype resolution failed for type " << typeid(T).name() << std::endl;
            return MPI_DATATYPE_NULL; 
        }

        template <>
        constexpr MPI_Datatype get_MPI_Datatype(double d) { return MPI_DOUBLE; }
        template <>
        constexpr MPI_Datatype get_MPI_Datatype(int i) { return MPI_INT; }

        template <>
        constexpr MPI_Datatype get_MPI_Datatype(size_t s) {
            static_assert( sizeof(size_t) == sizeof(int64_t) && sizeof(size_t) == sizeof(uint64_t) );
            return ( std::is_signed<size_t>() ? MPI_INT64_T : MPI_UINT64_T );
        }

        class state {

#if ENABLE_SHM
          private:
            MPI_Comm node_comm_;
#endif
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
#if ENABLE_SHM
                    prk::MPI::check( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm_) );
#endif
                }
            }

            ~state(void) {
                int is_init, is_final;
                MPI_Initialized(&is_init);
                MPI_Finalized(&is_final);
                if (is_init && !is_final) {
#if ENABLE_SHM
                    prk::MPI::check( MPI_Comm_free(&node_comm_) );
#endif
                    MPI_Finalize();
                }
            }

#if ENABLE_SHM
            MPI_Comm node_comm(void) {
                // this is a handle so we can always return a copy of the private instance
                return node_comm_;
            }
#endif
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

        void wait(MPI_Request req) {
            prk::MPI::check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
        }

        void waitall(MPI_Request * reqs, int count) {
            prk::MPI::check( MPI_Waitall(count, reqs, MPI_STATUS_IGNORE) );
        }

        void waitall(std::vector<MPI_Request> & reqs) {
            prk::MPI::check( MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUS_IGNORE) );
        }

        void barrier(MPI_Comm comm = MPI_COMM_WORLD) {
            prk::MPI::check( MPI_Barrier(comm) );
        }

        template <typename T>
        void bcast(T * buffer, int count = 1, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            //MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buffer);
            prk::MPI::check( MPI_Bcast(buffer, count * sizeof(T), MPI_BYTE, root, comm) );
        }

        template <typename T>
        void alltoall(const T * sbuffer, int scount, T * rbuffer, int rcount, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype stype = prk::MPI::get_MPI_Datatype(*sbuffer);
            MPI_Datatype rtype = prk::MPI::get_MPI_Datatype(*rbuffer);
            prk::MPI::check( MPI_Alltoall(sbuffer, scount, stype,
                                          rbuffer, rcount, rtype, comm) );
        }

        template <typename T>
        void alltoall(const std::vector<T> & sbuffer, std::vector<T> & rbuffer, MPI_Comm comm = MPI_COMM_WORLD) {
            int scount = sbuffer.size();
            int rcount = rbuffer.size();
            MPI_Datatype stype = prk::MPI::get_MPI_Datatype(*sbuffer.data());
            MPI_Datatype rtype = prk::MPI::get_MPI_Datatype(*rbuffer.data());
            prk::MPI::check( MPI_Alltoall(sbuffer.data(), scount, stype,
                                          rbuffer.data(), rcount, rtype, comm) );
        }

        template <typename T>
        void send(const T * buf, int dest, int count = 1, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buf);
            prk::MPI::check( MPI_Send(buf, count, dt, dest, tag, comm) );
        }

        template <typename T>
        void recv(T * buf, int source, int count = 1, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buf);
            prk::MPI::check( MPI_Recv(buf, count, dt, source, tag, comm, MPI_STATUS_IGNORE) );
        }

        template <typename T>
        void sendrecv(const T * sendbuf, int dest, T * recvbuf, int source, int count = 1, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*sendbuf);
            prk::MPI::check( MPI_Sendrecv(sendbuf, count, dt, dest, tag, recvbuf, count, dt, source, tag, comm, MPI_STATUS_IGNORE) );
        }

        template <typename T>
        MPI_Request isend(const T * buf, int dest, int count = 1, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buf);
            MPI_Request req = MPI_REQUEST_NULL;
            prk::MPI::check( MPI_Isend(buf, count, dt, dest, tag, comm, &req) );
            return req;
        }

        template <typename T>
        MPI_Request irecv(T * buf, int source, int count = 1, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buf);
            MPI_Request req = MPI_REQUEST_NULL;
            prk::MPI::check( MPI_Irecv(buf, count, dt, source, tag, comm, &req) );
            return req;
        }

        template <typename T>
        MPI_Request rget(MPI_Win win, T * buf, int target_rank, size_t target_offset, int count = 1) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buf);
            MPI_Request req = MPI_REQUEST_NULL;
            prk::MPI::check( MPI_Rget(buf, count, dt, target_rank, target_offset, count, dt, win, &req) );
            return req;
        }

        template <typename T>
        void bget(MPI_Win win, T * buf, int target_rank, size_t target_offset, int count = 1) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(*buf);
            MPI_Request req = MPI_REQUEST_NULL;
            prk::MPI::check( MPI_Rget(buf, count, dt, target_rank, target_offset, count, dt, win, &req) );
            prk::MPI::wait(req);
        }

        template <typename T>
        std::pair<MPI_Win, T*> win_allocate(size_t count, int disp_unit = sizeof(T), MPI_Info info = MPI_INFO_NULL, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Win win = MPI_WIN_NULL;
            T * buffer = nullptr;
            prk::MPI::check( MPI_Win_allocate(count * sizeof(T), disp_unit, info, comm, &buffer, &win) );
            //std::cerr << "MPI_Win_allocate buffer = " << buffer << std::endl;
            prk::MPI::check( MPI_Win_lock_all(MPI_MODE_NOCHECK, win) );
            return {win,buffer};
        }

        void win_free(MPI_Win win) {
            prk::MPI::check( MPI_Win_unlock_all(win) );
            prk::MPI::check( MPI_Win_free(&win) );
        }

        void win_sync(MPI_Win win) {
            prk::MPI::check( MPI_Win_sync(win) );
        }

        template <typename T>
        T min(T in, MPI_Comm comm = MPI_COMM_WORLD) {
            T out;
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(in);
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, dt, MPI_MIN, comm) );
            return out;
        }

        template <typename T>
        T max(T in,  MPI_Comm comm = MPI_COMM_WORLD) {
            T out;
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(in);
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, dt, MPI_MAX, comm) );
            return out;
        }

        template <typename T>
        T sum(T in, MPI_Comm comm = MPI_COMM_WORLD) {
            T out;
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(in);
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, dt, MPI_SUM, comm) );
            return out;
        }

        template <typename T>
        T avg(T in, MPI_Comm comm = MPI_COMM_WORLD) {
            T out;
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(1);
            prk::MPI::check( MPI_Allreduce(&in, &out, 1, dt, MPI_SUM, comm) );
            out /= prk::MPI::size(comm);
            return out;
        }

        template <typename T>
        void stats(T in, T * min, T * max, T * avg, MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(in);
            prk::MPI::check( MPI_Allreduce(&in, min, 1, dt, MPI_MIN, comm) );
            prk::MPI::check( MPI_Allreduce(&in, max, 1, dt, MPI_MAX, comm) );
            prk::MPI::check( MPI_Allreduce(&in, avg, 1, dt, MPI_SUM, comm) );
            *avg /= prk::MPI::size(comm);
        }

        template <typename T>
        bool is_same(T in, MPI_Comm comm = MPI_COMM_WORLD) {
            T min=std::numeric_limits<T>::max();
            T max=std::numeric_limits<T>::min();
            MPI_Datatype dt = prk::MPI::get_MPI_Datatype(in);
            prk::MPI::check( MPI_Allreduce(&in, &min, 1, dt, MPI_MIN, comm) );
            prk::MPI::check( MPI_Allreduce(&in, &max, 1, dt, MPI_MAX, comm) );
            return (min==max);
        }

        bool is_same(size_t in, MPI_Comm comm = MPI_COMM_WORLD) {
            size_t min=SIZE_MAX, max=0;
            static_assert( sizeof(size_t) == sizeof(int64_t) && sizeof(size_t) == sizeof(uint64_t) );
            MPI_Datatype dt = (std::is_signed<size_t>() ? MPI_INT64_T : MPI_UINT64_T);
            prk::MPI::check( MPI_Allreduce(&in, &min, 1, dt, MPI_MIN, comm) );
            prk::MPI::check( MPI_Allreduce(&in, &max, 1, dt, MPI_MAX, comm) );
            return (min==max);
        }

        size_t sum(size_t in, MPI_Comm comm = MPI_COMM_WORLD) {
            size_t out;
            static_assert( sizeof(size_t) == sizeof(int64_t) && sizeof(size_t) == sizeof(uint64_t) );
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
                if ((size_t)me_ < remainder) local_size_++;

                {
                    MPI_Datatype dt = (std::is_signed<size_t>() ? MPI_INT64_T : MPI_UINT64_T);
                    std::vector<size_t> global_sizes(np_);   // in
                    global_offsets_.resize(np_);             // out
                    // there is probably a better way to do this.  i should be able to MPI_Exscan then MPI_Allgather instead.
                    prk::MPI::check( MPI_Allgather(&local_size_, 1, dt, global_sizes.data(), 1, dt, comm_) );
#if 0
                    std::exclusive_scan( global_sizes.cbegin(), global_sizes.cend(), global_offsets_.begin(), 0);
#else
                    global_offsets_[0] = 0;
                    for ( size_t i = 1 ; i < global_sizes.size() ; ++i ) {
                        global_offsets_[i] = global_sizes[i-1];

                    }
#endif
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

            T * data(void) noexcept
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
                        MPI_Request req = MPI_REQUEST_NULL;
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
                        MPI_Request req = MPI_REQUEST_NULL;
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
                        MPI_Request req = MPI_REQUEST_NULL;
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

        template <typename TF, typename TI>
        void print_matrix(const TF * matrix, TI rows, TI cols, const std::string label = "") {
            int me = prk::MPI::rank();
            int np = prk::MPI::size();

            //std::cout << "@" << me << " rows=" << rows << " cols=" << cols << std::endl;

            //std::cerr << std::endl;
            prk::MPI::barrier();

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
                prk::MPI::barrier();
            }
            //prk::MPI::barrier();
        }

        template <typename TF, typename TI>
        void print_matrix(const prk::vector<TF> & matrix, TI rows, TI cols, const std::string label = "") {
            int me = prk::MPI::rank();
            int np = prk::MPI::size();

            //std::cout << "@" << me << " rows=" << rows << " cols=" << cols << std::endl;

            //std::cerr << std::endl;
            prk::MPI::barrier();

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
                prk::MPI::barrier();
            }
            //prk::MPI::barrier();
        }

    } // MPI namespace

} // prk namespace

#endif // PRK_MPI_HPP
