#ifndef PRK_MPI_HPP
#define PRK_MPI_HPP

#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

namespace prk
{
    namespace MPI
    {
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

                MPI_Abort(MPI_COMM_WORLD, errorcode);
                std::abort(); // unreachable
            }
        }

        class state {

          public:
            state(void) {
                int is_init, is_final;
                MPI_Initialized(&is_init);
                MPI_Finalized(&is_final);
                if (!is_init && !is_final) {
                    MPI_Init(NULL,NULL);
                }
            }

            state(int argc, char** argv) {
                int is_init, is_final;
                MPI_Initialized(&is_init);
                MPI_Finalized(&is_final);
                if (!is_init && !is_final) {
                    MPI_Init(&argc,&argv);
                }
            }

            ~state(void) {
                int is_init, is_final;
                MPI_Initialized(&is_init);
                MPI_Finalized(&is_final);
                if (is_init && !is_final) {
                    MPI_Finalize();
                }
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

    } // MPI namespace

} // prk namespace

#endif // PRK_MPI_HPP
