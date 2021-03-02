/// https://github.com/codeplaysoftware/computecpp-sdk/blob/master/samples/matrix-multiply.cpp

/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  matrix-multiply.cpp
 *
 *  Description:
 *    Example of matrix multiplication in SYCL.
 *
 **************************************************************************/

using namespace cl::sycl;

template <typename T> class dgemm_tiled;

template <typename T>
void prk_dgemm(sycl::queue & q,
               int order, int tile_size,
               sycl::buffer<T> & d_A,
               sycl::buffer<T> & d_B,
               sycl::buffer<T> & d_C)
{
    q.submit([&](handler& cgh)
    {
        auto pA = d_A.template get_access<access::mode::read>(cgh);
        auto pB = d_B.template get_access<access::mode::read>(cgh);
        auto pC = d_C.template get_access<access::mode::read_write>(cgh);

        auto localRange = range<1>(tile_size * tile_size);

        accessor<T, 1, access::mode::read_write, access::target::local> pBA( localRange, cgh);
        accessor<T, 1, access::mode::read_write, access::target::local> pBB( localRange, cgh);

        cgh.parallel_for<class dgemm_tiled<T>>(
            nd_range<2>{range<2>(order, order),
                        range<2>(tile_size, tile_size)},
            [=](nd_item<2> it) {

              // Current block
              int blockX = it.get_group(1);
              int blockY = it.get_group(0);

              // Current local item
              int localX = it.get_local_id(1);
              int localY = it.get_local_id(0);

              // Start in the A matrix
              int a_start = order * tile_size * blockY;
              // End in the b matrix
              int a_end = a_start + order - 1;
              // Start in the b matrix
              int b_start = tile_size * blockX;

              // Result for the current C(i,j) element
              T tmp(0);

              // We go through all a, b blocks
              for (int a = a_start, b = b_start; a <= a_end; a += tile_size, b += (tile_size * order)) {

                // Copy the values in shared memory collectively
                pBA[localY * tile_size + localX] = pA[a + order * localY + localX];
                // Note the swap of X/Y to maintain contiguous access
                pBB[localX * tile_size + localY] = pB[b + order * localY + localX];

                it.barrier(access::fence_space::local_space);

                // Now each thread adds the value of its sum
                for (int k = 0; k < tile_size; k++) {
                  tmp += pBA[localY * tile_size + k] * pBB[localX * tile_size + k];
                }

                // The barrier ensures that all threads have written to local memory before continuing
                it.barrier(access::fence_space::local_space);
              }
              auto elemIndex = it.get_global_id(0) * it.get_global_range()[1] +
                               it.get_global_id(1);

              // Each thread updates its position
              pC[elemIndex] += tmp;
        });
    });
    q.wait();
}

