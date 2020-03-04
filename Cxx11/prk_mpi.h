#ifndef PRK_MPI_HPP
#define PRK_MPI_HPP

#include <iostream>
#include <vector>
#include <string>

#include <cinttypes>
#include <type_traits>

#include <mpi.h>

namespace prk
{
    namespace MPI
    {
        void abort(int errorcode = -1)
        {
            MPI_Abort(MPI_COMM_WORLD, errorcode);
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
              MPI_Win shm_win_;
              MPI_Win distributed_win_;
              T * local_pointer;

          public:
            vector(size_t global_size, MPI_Comm comm = MPI_COMM_WORLD)
            {
                int np = prk::MPI::size(comm);
                int me = prk::MPI::rank(comm);

                bool consistency = prk::MPI::is_same(global_size, comm);
                if (!consistency) {
                    if (me == 0) std::cerr << "global size inconsistent!\n"
                                           << " rank = " << me << ", global size = " << global_size << std::endl;
                    prk::MPI::abort();
                }

                size_t local_size = global_size / np;
                size_t remainder  = global_size % np;
                if (me < remainder) local_size++;

                size_t verify_global_size = sum(local_size, comm);
                if (global_size != verify_global_size) {
                    if (me == 0) std::cerr << "global size inconsistent!\n"
                                           << " expected: " << global_size << "\n"
                                           << " actual: " << verify_global_size << "\n";
                    std::cerr << "rank = " << me << ", local size = " << local_size << std::endl;
                    prk::MPI::abort();
                }

                MPI_Comm node_comm;
                prk::MPI::check( MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm) );

                prk::MPI::check( MPI_Win_allocate_shared(local_size, 1, MPI_INFO_NULL, node_comm,
                                                         &this->local_pointer, &this->shm_win_) );

                prk::MPI::check( MPI_Win_allocate_shared(local_size, 1, MPI_INFO_NULL, comm,
                                                         &this->local_pointer, &this->distributed_win_) );

                prk::MPI::check( MPI_Comm_free(&node_comm) );


            }
        };

    } // MPI namespace

} // prk namespace

#endif // PRK_MPI_HPP
