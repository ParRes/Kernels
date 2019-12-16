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
            if (rc==MPI_SUCCESS) {
                return;
            } else {
                int resultlen;

                std::string errorcode_string(MPI_MAX_ERROR_STRING);
                std::string errorclass_string(MPI_MAX_ERROR_STRING);

                int errorclass;
                MPI_Error_class(errorcode, &errorclass);
                int rc = MPI_Error_string(errorclass, errorclass_string.data(), &resultlen);
                std::cerr << "MPI error: class " << errorclass << ", " << errorclass_string << std::endl;

                int rc = MPI_Error_string(errorcode, errorcode_string.data(), &resultlen);
                std::cerr << "MPI error: code " << errorcode << ", " << errorcode_string << std::endl;

                MPI_Abort(MPI_COMM_WORLD, errorcode);
                std::abort(); // unreachable
            }
        }

        class state {

            private:
                int world_size;
                int world_rank;

            public:
                int me(void) { return world_rank; }
                int np(void) { return world_size; }


            state(void) {
                MPI_Init(NULL,NULL);
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            }

            state(int argc, char** argv) {
                MPI_Init(&argc, &argv);
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            }

            ~state(void) {
                MPI_Finalize();
            }

        };

        void barrier(void) {
            prk::MPI::check( MPI_Barrier(MPI_COMM_WORLD) );
        }

        void barrier(MPI_Comm comm) {
            prk::MPI::check( MPI_Barrier(comm) );
        }


    } // MPI namespace

} // prk namespace

#endif // PRK_MPI_HPP
