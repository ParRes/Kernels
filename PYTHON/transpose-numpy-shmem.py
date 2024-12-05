#!/usr/bin/env python3
#
# Copyright (c) 2020, Intel Corporation
# Copyright (c) 2023, NVIDIA
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

#*******************************************************************
#
# NAME:    transpose
#
# PURPOSE: This program measures the time for the transpose of a
#          column-major stored matrix into a row-major stored matrix.
#
# USAGE:   Program input is the matrix order and the number of times to
#          repeat the operation:
#
#          transpose <# iterations> <matrix_size>
#
#          The output consists of diagnostics to make sure the
#          transpose worked and timing statistics.
#
# HISTORY: Written by  Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#
# *******************************************************************

#                     Layout nomenclature
#                     -------------------
#
# - Each rank owns one block of columns (Colblock) of the overall
#   matrix to be transposed, as well as of the transposed matrix.
# - Colblock is stored contiguously in the memory of the rank.
#   The stored format is column major, which means that matrix
#   elements (i,j) and (i+1,j) are adjacent, and (i,j) and (i,j+1)
#   are "order" words apart
# - Colblock is logically composed of #ranks Blocks, but a Block is
#   not stored contiguously in memory. Conceptually, the Block is
#   the unit of data that gets communicated between ranks. Block i of
#   rank j is locally transposed and gathered into a buffer called Work,
#   which is sent to rank i, where it is scattered into Block j of the
#   transposed matrix.
# - When tiling is applied to reduce TLB misses, each block gets
#   accessed by tiles.
# - The original and transposed matrices are called A and B
#
# +-----------------------------------------------------------------+
# |           |           |           |                             |
# | Colblock  |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |        -------------------------------                          |
# |           |           |           |                             |
# |           |  Block    |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |        -------------------------------                          |
# |           |           |           |                             |
# |           |           |           |   Overall Matrix            |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |        -------------------------------                          |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# +-----------------------------------------------------------------+

import sys
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer
from shmem4py import shmem
import numpy

def main():

    me = shmem.my_pe()
    np = shmem.n_pes()

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    if (me==0):
        print('Parallel Research Kernels version ') #, PRKVERSION
        print('Python SHMEM/Numpy  Matrix transpose: B = A^T')

    if len(sys.argv) != 3:
        if (me==0):
            print('argument count = ', len(sys.argv))
            print("Usage: ./transpose <# iterations> <matrix order>")
        sys.exit()

    iterations = int(sys.argv[1])
    if iterations < 1:
        if (me==0):
            print("ERROR: iterations must be >= 1")
        sys.exit()

    order = int(sys.argv[2])
    if order < 1:
        if (me==0):
            print("ERROR: order must be >= 1")
        sys.exit()

    if order % np != 0:
        if (me==0):
            print(f"ERROR: matrix order ({order}) should be divisible by # procs ({np})")
        sys.exit()

    block_order = int(order / np)

    if (me==0):
        print('Number of ranks      = ', np)
        print('Number of iterations = ', iterations)
        print('Matrix order         = ', order)

    shmem.barrier_all()

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    LA = numpy.fromfunction(lambda i,j:  me * block_order + i*order + j, (order,block_order), dtype='d')
    A = shmem.full((order,block_order),LA)
    B = shmem.zeros((order,block_order))
    T = shmem.zeros((order,block_order))

    for k in range(0,iterations+1):

        if k<1:
            shmem.barrier_all()
            t0 = timer()

        # this actually forms the transpose of A
        #B += numpy.transpose(A)
        # this only uses the transpose _view_ of A
        #B += A.T

        # barrier required before alltoall for correctness
        shmem.barrier_all()
        shmem.alltoall(T, A)
        for r in range(0,np):
            lo = block_order * r
            hi = block_order * (r+1)
            #B[lo:hi,:] += numpy.transpose(T[lo:hi,:])
            B[lo:hi,:] += T[lo:hi,:].T

        A += 1.0

    shmem.barrier_all()
    t1 = timer()
    trans_time = t1 - t0

    shmem.free(A)
    shmem.free(T)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    # allgather is non-scalable but was easier to debug
    F = shmem.zeros((np,order,block_order))
    shmem.fcollect(F,B)
    G = numpy.concatenate(F,axis=1)
    #if (me==0):
    #    print(G)
    H = numpy.fromfunction(lambda i,j: ((iterations/2.0)+(order*j+i))*(iterations+1.0), (order,order), dtype='d')
    abserr = numpy.linalg.norm(numpy.reshape(G-H,order*order),ord=1)

    shmem.free(B)
    shmem.free(F)

    epsilon=1.e-8
    nbytes = 2 * order**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if abserr < epsilon:
        if (me==0):
            print('Solution validates')
            avgtime = trans_time/iterations
            print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime)
    else:
        if (me==0):
            print('error ',abserr, ' exceeds threshold ',epsilon)
            print("ERROR: solution did not validate")


if __name__ == '__main__':
    main()
