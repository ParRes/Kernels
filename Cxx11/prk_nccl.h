#ifndef PRK_NCCL_HPP
#define PRK_NCCL_HPP

#include <iostream>
#include <vector>
#include <array>

#include <nccl.h>

namespace prk
{
    void check(ncclResult_t rc)
    {
        if (rc!=ncclSuccess) {
            std::cerr << "PRK NCCL error: " << ncclGetErrorString(rc) << std::endl;
            std::abort();
        }
    }

    namespace NCCL
    {

    } // NCCL namespace

} // prk namespace

#endif // PRK_NCCL_HPP
