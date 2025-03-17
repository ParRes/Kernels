#ifndef PRK_NCCL_HPP
#define PRK_NCCL_HPP

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <typeinfo>

#include <nccl.h>

#include "prk_cuda.h"

namespace prk
{
    void check(ncclResult_t rc)
    {
        if (rc != ncclSuccess) {
            std::cerr << "PRK NCCL error: " << ncclGetErrorString(rc) << std::endl;
            std::abort();
        }
    }

    namespace NCCL
    {

        template <typename T>
        ncclDataType_t get_NCCL_Datatype(T t) { 
            std::cerr << "get_NCCL_Datatype resolution failed for type " << typeid(T).name() << std::endl;
            std::abort();
        }

        template <>
        constexpr ncclDataType_t get_NCCL_Datatype(double d) { return ncclFloat64; }
        template <>
        constexpr ncclDataType_t get_NCCL_Datatype(int i) { return ncclInt32; }

        template <typename T>
        void alltoall(const T * sbuffer, T * rbuffer, size_t count, ncclComm_t comm, cudaStream_t stream = 0) {
            ncclDataType_t type = get_NCCL_Datatype(*sbuffer);

            int np;
            prk::check( ncclCommCount(comm, &np) );

            prk::check( ncclGroupStart() );
            for (int r=0; r<np; r++) {
                prk::check( ncclSend(sbuffer + r*count, count, type, r, comm, stream) );
                prk::check( ncclRecv(rbuffer + r*count, count, type, r, comm, stream) );
            }
            prk::check( ncclGroupEnd() );
        }

        template <typename T>
        void sendrecv(const T * sbuffer,  int dst, T * rbuffer, int src, size_t count, ncclComm_t comm, cudaStream_t stream = 0) {
            ncclDataType_t type = get_NCCL_Datatype(*sbuffer);
            prk::check( ncclGroupStart() );
            {
                prk::check( ncclSend(sbuffer, count, type, dst, comm, stream) );
                prk::check( ncclRecv(rbuffer, count, type, src, comm, stream) );
            }
            prk::check( ncclGroupEnd() );
        }

    } // NCCL namespace

} // prk namespace

#endif // PRK_NCCL_HPP
