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
from mpi4py import MPI
import numpy

def main():
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    np = comm.Get_size()

    final = np-1

    if me==0:
        print("Parallel Research Kernels")
        print("MPI pipeline execution on 2D grid")

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(f"argument count = {len(sys.argv)}")
        sys.exit("Usage: ... <# iterations> <1st array dimension> <2nd array dimension> [<group factor>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")
    m = int(sys.argv[2])
    if m < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    n = int(sys.argv[3])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    if len(sys.argv) == 5:
        grp = int(sys.argv[4])
        if grp < 1:
            sys.exit("ERROR: group factor must be >= 1")
    else:
        grp = 1

    if me == 0:
        print(f"Number of ranks                = {np}")
        print(f"Grid sizes                     = {m}, {n}")
        print(f"Number of iterations           = {iterations}")
        if (grp > 1):
            print(f"Group factor                   = {grp}")

    local_pipeline_time = 0.0
    pipeline_time = 0.0

    segment_size = m//np
    leftover     = m%np
    if me < leftover:
        start = (segment_size+1) * me
        end   = start + segment_size
    else:
        start = (segment_size+1) * leftover + segment_size * (me-leftover)
        end = start + segment_size - 1

    # now set segment_size to the value needed by the calling rank
    segment_size = end - start + 1
    grid = numpy.zeros((segment_size+1,n), dtype='d')

    inbuf = numpy.zeros(grp, dtype='d')
    outbuf = numpy.zeros(grp, dtype='d')

    # set boundary values (bottom and left side of grid)
    if me==0:
        grid[0,:] = list(range(n))
    for i in range(start-1,end+1):
        grid[i-start,0] = i

    # redefine start and end for calling rank to reflect local indices
    if me==0:
        start = 1
    else:
        start = 0
    end = segment_size-1

    for iter in range(0,iterations+1):
        if iter == 1:
            comm.Barrier()
            local_pipeline_time = MPI.Wtime()

        # special case for no grouping
        if grp == 1:
            for j in range(1,n):
                # if I am not at the left boundary, I need to wait for my left neighbor to send data
                if me > 0:
                    comm.Recv(grid[start-1,j:j+1], source=me-1, tag=j)

                for i in range(start,end+1):
                    grid[i,j] = grid[i-1,j] + grid[i,j-1] - grid[i-1,j-1]

                # if I am not on the right boundary, send data to my right neighbor
                if me < np-1:
                    comm.Send(grid[end,j:j+1], dest=me+1, tag=j)

        # apply grouping
        else:
            for j in range(1, n, grp):
                jjsize = min(grp, n-j)

                # if I am not at the left boundary, I need to wait for my left neighbor to send data
                if me > 0:
                    comm.Recv(inbuf, source=me-1, tag=j)
                    grid[start-1,j:j+jjsize] = inbuf[0:jjsize]

                for jj in range(0, jjsize):
                    for i in range(start, end+1):
                        grid[i,jj+j] = grid[i-1,jj+j] + grid[i,jj+j-1] - grid[i-1,jj+j-1]

                # if I am not on the right boundary, send data to my right neighbor
                if me < np-1:
                    outbuf[0:jjsize] = grid[end,j:j+jjsize]
                    comm.Send(outbuf, dest=me+1, tag=j)

        # copy top right corner value to bottom left corner to create dependency
        if np > 1:
            if me == final:
                corner_val = -grid[end,n-1]
                comm.Send(corner_val, dest=0, tag=888)

            if me == 0:
                comm.Recv(grid[0,0:1], source=final, tag=888)
        else:
            grid[0,0] = -grid[end,n-1]

    local_pipeline_time = MPI.Wtime() - local_pipeline_time
    pipeline_time = comm.reduce(local_pipeline_time, op=MPI.MAX, root=final)

    # verify correctness, using top right value
    corner_val = (iterations+1)*(m+n-2)
    if me == final:
        epsilon = 1e-8
        if abs(grid[end,n-1]-corner_val)/corner_val >= epsilon:
            print(f"ERROR: checksum {grid[end,n-1]} does not match verification value {corner_val}")
            sys.exit()

    if me == final:
        avgtime = pipeline_time/iterations
        print(f"Solution validates; verification value = {corner_val}")
        print(f"Rate (MFlops/s): {1e-6 * 2 * (((m-1)*(n-1)))/avgtime} Avg time (s): {avgtime}")


if __name__ == '__main__':
    main()
