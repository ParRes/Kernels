#ifndef PRK_CUDA_HPP
#define PRK_CUDA_HPP

#include <cstdio>
#include <cstdlib>

#ifndef __NVCC__
#warning Please compile CUDA code with CC=nvcc.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif

namespace prk
{
    void CUDAinfo()
    {
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Device name: " << prop.name << "\n";
#ifndef __CORIANDERCC__
            std::cout << "Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n";
            std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
#endif
        }
    }

    inline void CUDAcheck(cudaError_t rc)
    {
        if (rc==cudaSuccess) {
            return;
        } else {
            std::cerr << "PRK CUDA error: " << cudaGetErrorString(rc) << std::endl;
            std::abort();
        }
    }
}

#endif // PRK_CUDA_HPP
