#!/usr/bin/env python3
#
# Copyright (c) 2023
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# *******************************************************************
#
# NAME:    Pipeline
#
# PURPOSE: This program tests the efficiency with which point-to-point
#          synchronization can be carried out. It does so by executing
#          a pipelined algorithm on an m*n grid. The first array dimension
#          is distributed among the ranks (stripwise decomposition).
#
# USAGE:   The program takes as input the dimensions of the grid, and the
#          number of times we loop over the grid
#
#                <progname> <# iterations> <m> <n>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Rob Van der Wijngaart, March 2006.
#          - Modified by Rob Van der Wijngaart, August 2006:
#             * changed boundary conditions and stencil computation to avoid
#               overflow
#             * introduced multiple iterations over grid and dependency between
#               iterations
#          - Converted to Python by Marcin Rogowski, May 2023.
#
# *******************************************************************

import sys
import time
import numpy
from shmem4py import shmem

def main():
    me = shmem.my_pe()
    np = shmem.n_pes()

    root = np-1

    if me==0:
        print("Parallel Research Kernels")
        print("SHMEM pipeline execution on 2D grid")

    if len(sys.argv) != 4:
        print(f"argument count = {len(sys.argv)}")
        sys.exit("Usage: ... <# iterations> <1st array dimension> <2nd array dimension>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    m = int(sys.argv[2])
    if m < 1:
        sys.exit("ERROR: array dimension must be >= 1")
    if m <= np:
        print("Error: m must be greater than the number of PEs")
        exit(1)

    n = int(sys.argv[3])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    if me == root:
        print(f"Number of ranks            = {np}")
        print(f"Grid sizes                 = {m}, {n}")
        print(f"Number of iterations       = {iterations}")
        print(f"No handshake between neighbor threads")

    shmem.barrier_all()

    dst = shmem.zeros(n, dtype='d')
    src = numpy.zeros(n, dtype='d')

    flag_left = shmem.zeros(n, dtype='i')

    local_pipeline_time = shmem.zeros(1, dtype='d')
    pipeline_time       = shmem.zeros(1, dtype='d')

    start = numpy.zeros(np, dtype='i')
    end   = numpy.zeros(np, dtype='i')

    for i in range(0,np):
        segment_size = m//np
        if i < m%np:
            segment_size += 1
        if i > 0:
            start[i] = end[i-1]+1

        end[i] = start[i]+segment_size-1

    segment_size = end[me] - start[me] + 1
    grid = numpy.zeros((segment_size+1,n), dtype='d')

    # set boundary values (bottom and left side of grid)
    if me==0:
        grid[0,:] = list(range(n))
    for i in range(start[me]-1,end[me]+1):
        grid[i-start[me],0] = i

    # redefine start and end for calling rank to reflect local indices
    if me==0:
        start[me] = 1
    else:
        start[me] = 0
    end[me] = segment_size-1

    # initialize synchronization flags
    true = shmem.array([1], dtype='i')
    false = shmem.array([0], dtype='i')

    shmem.barrier_all()

    for iter in range(0,iterations+1):
        true[0] = (iter+1)%2
        false[0] = 0 if true[0] else 1

        if iter == 1:
            shmem.barrier_all()
            local_pipeline_time[0] = time.monotonic()

        if me==0 and np>1:
            shmem.wait_until(flag_left[0:1], shmem.CMP.EQ, false)
            if iter>0:
                grid[start[me]-1,0] = dst[0]

        for j in range(1,n):
            if me > 0:
                shmem.wait_until(flag_left[j:j+1], shmem.CMP.EQ, true)
                grid[start[me]-1,j] = dst[j]

            for i in range(start[me],end[me]+1):
                grid[i,j] = grid[i-1,j] + grid[i,j-1] - grid[i-1,j-1]

            if me != np-1:
                src[j] = grid[end[me],j]

                shmem.put(dst[j:j+1], src[j:j+1], me+1)
                shmem.fence()

                # indicate to right neighbor that data is available
                shmem.put(flag_left[j:j+1], true, me+1)

        if np > 1:
            if me == root:
                corner_val = -grid[end[me],n-1]
                src [0] = corner_val
                shmem.put(dst[0:1], src[0:1], 0)
                shmem.fence()
                # indicate to PE 0 that data is available
                shmem.put(flag_left[0:1], true, 0)
        else:
            grid[0,0] = -grid[end[me],n-1]

    local_pipeline_time[0] = time.monotonic() - local_pipeline_time[0]
    shmem.max_reduce(pipeline_time, local_pipeline_time)

    # verify correctness, using top right value
    corner_val = (iterations+1)*(m+n-2)
    if me == root:
        epsilon = 1e-8
        if abs(grid[end[me],n-1]-corner_val)/corner_val >= epsilon:
            print(f"ERROR: checksum {grid[end[me],n-1]} does not match verification value {corner_val}")
            shmem.global_exit(1)

    if me == root:
        avgtime = pipeline_time[0]/iterations
        print(f"Solution validates; verification value = {corner_val}")
        print(f"Rate (MFlops/s): {1e-6 * 2 * (((m-1)*(n-1)))/avgtime} Avg time (s): {avgtime}")


if __name__ == '__main__':
    main()
