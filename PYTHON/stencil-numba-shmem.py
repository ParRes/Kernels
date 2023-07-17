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
# NAME:    Stencil
#
# PURPOSE: This program tests the efficiency with which a space-invariant,
#          linear, symmetric filter (stencil) can be applied to a square
#          grid or image.
#
# USAGE:   The program takes as input the linear
#          dimension of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <grid size>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Tom St. John, July 2015.
#          - Adapted by Rob Van der Wijngaart to introduce double buffering, December 2015
#          - Converted to Python (MPI) by Jeff Hammond, February 2016
#          - Converted to Python (SHMEM) by Marcin Rogowski, May 2023
#
# *******************************************************************

# TODO: currently, only the star stencil is implemented

import sys
import time
import numpy
from shmem4py import shmem
from numba import jit

@jit(nopython=True)
def star(n, r, A, B, W, jstart, jend, istart, iend):
    for a in range(max(jstart, r), min(n-r-1, jend)+1):
        a = a - jstart
        for b in range(max(istart, r), min(n-r-1, iend)+1):
            b = b - istart

            for k in range(2 * r + 1):
                B[a,b] += W[r,k] * A[a + k, b + r]
                B[a,b] += W[k,r] * A[a + r, b + k]

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r % fac1 == 0:
            fac2 = r//fac1
            break
    return fac1, fac2

def main():
    splitfence = False

    me = shmem.my_pe()
    np = shmem.n_pes()

    if me==0:
        print("Parallel Research Kernels")
        print("Python SHMEM/Numba Stencil execution on 2D grid")

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        if (me==0):
            print(f"argument count = {len(sys.argv)}")
            print("Usage: ... <# iterations> <array dimension> [<radius>]")
        sys.exit()

    iterations = int(sys.argv[1])
    if iterations < 1:
        if (me==0):
            print("ERROR: iterations must be >= 1")
        sys.exit()

    n = int(sys.argv[2])
    nsquare = n * n
    if nsquare < np:
        if (me==0):
            print(f"ERROR: grid size {nsquare} must be at least # ranks: {np}")
        sys.exit()

    if len(sys.argv) > 3:
        pattern = sys.argv[3]
    else:
        pattern = 'star'

    if pattern != 'star':
        if (me==0):
            print("ERROR: Only star pattern is supported")
        sys.exit()

    if len(sys.argv) > 4:
        radius = int(sys.argv[4])
        if radius < 1:
            if (me==0):
                print("ERROR: Stencil radius should be positive")
            sys.exit()
        if 2*radius+1 > n:
            if (me==0):
                print("ERROR: Stencil radius exceeds grid size")
            sys.exit()
    else:
        radius = 2

    if me == 0:
        print("Number of ranks      = ", np)
        print("Number of iterations = ", iterations)
        print("Grid size            = ", n)
        print("Type of stencil      = ", pattern)
        print("Radius of stencil    = ", radius)
        print("Data type            = float 64 (double precision in C)")

    weight = numpy.zeros((2*radius+1, 2*radius+1), dtype='d')

    local_stencil_time = shmem.zeros(1, dtype='d')
    stencil_time       = shmem.zeros(1, dtype='d')
    local_norm = shmem.zeros(1, dtype='d')
    norm       = shmem.zeros(1, dtype='d')
    iterflag   = shmem.zeros(2, dtype='i')
    width      = shmem.zeros(1, dtype='i')
    maxwidth   = shmem.zeros(1, dtype='i')
    height     = shmem.zeros(1, dtype='i')
    maxheight  = shmem.zeros(1, dtype='i')

    npx, npy = factor(np)

    mex = me%npx
    mey = me//npx
    right_nbr  = me+1
    left_nbr   = me-1
    top_nbr    = me+npx
    bottom_nbr = me-npx
    count_case = 4

    if mex == 0:
        count_case -= 1
    if mex == npx-1:
        count_case -= 1
    if mey == 0:
        count_case -= 1
    if mey == npy-1:
        count_case -= 1

    shmem.barrier_all()

    width[0] = n//npx
    leftover = n%npx
    if mex < leftover:
        istart = (width[0]+1) * mex
        iend = istart + width[0] + 1
    else:
        istart = (width[0]+1) * leftover + width[0] * (mex-leftover)
        iend = istart + width[0]

    width[0] = iend - istart + 1
    if width[0] == 0:
        print(f"ERROR: rank {me} has no work to do")
        shmem.global_exit(1)

    height[0] = n//npy
    leftover = n%npy
    if mey < leftover:
        jstart = (height[0]+1) * mey
        jend = jstart + height[0] + 1
    else:
        jstart = (height[0]+1) * leftover + height[0] * (mey-leftover)
        jend = jstart + height[0]

    height[0] = jend - jstart + 1
    if height[0] == 0:
        print(f"ERROR: rank {me} has no work to do")
        shmem.global_exit(1)

    if width[0] < radius or height[0] < radius:
        print(f"ERROR: rank {me} has work tile smaller then stencil radius")
        shmem.global_exit(1)

    a = numpy.fromfunction(lambda i, j: i+istart+j+jstart, (height[0], width[0]), dtype='d')
    A = numpy.zeros((height[0]+2*radius, width[0]+2*radius), dtype='d')
    A[radius:-radius, radius:-radius] = a
    B = numpy.zeros((height[0], width[0]), dtype='d')

    shmem.barrier_all()
    shmem.max_reduce(maxwidth, width)
    shmem.barrier_all()
    shmem.max_reduce(maxheight, height)

    for ii in range(1, radius+1):
        weight[0+radius][ii+radius] = 1.0/(2.0*ii*radius)
        weight[ii+radius][0+radius] = 1.0/(2.0*ii*radius)
        weight[0+radius][-ii+radius] = -1.0/(2.0*ii*radius)
        weight[-ii+radius][0+radius] = -1.0/(2.0*ii*radius)

    # allocate communication buffers for halo values
    top_buf_out    = shmem.zeros(radius*maxwidth[0], dtype='d')
    bottom_buf_out = shmem.zeros(radius*maxwidth[0], dtype='d')

    top_buf_in    = {}
    bottom_buf_in = {}
    top_buf_in[0]    = shmem.zeros(radius*maxwidth[0], dtype='d')
    top_buf_in[1]    = shmem.zeros(radius*maxwidth[0], dtype='d')
    bottom_buf_in[0] = shmem.zeros(radius*maxwidth[0], dtype='d')
    bottom_buf_in[1] = shmem.zeros(radius*maxwidth[0], dtype='d')

    right_buf_out = shmem.zeros(radius*maxheight[0], dtype='d')
    left_buf_out  = shmem.zeros(radius*maxheight[0], dtype='d')

    right_buf_in = {}
    left_buf_in  = {}
    right_buf_in[0] = shmem.zeros(radius*maxheight[0], dtype='d')
    right_buf_in[1] = shmem.zeros(radius*maxheight[0], dtype='d')
    left_buf_in[0]  = shmem.zeros(radius*maxheight[0], dtype='d')
    left_buf_in[1]  = shmem.zeros(radius*maxheight[0], dtype='d')

    shmem.barrier_all()

    for iter in range(0, iterations+1):
        # start timer after a warmup iteration
        if iter == 1:
            shmem.barrier_all()
            local_stencil_time[0] = time.monotonic()

        # sw determines which incoming buffer to select
        sw = iter % 2

        # need to fetch ghost point data from neighbors
        if mey < npy-1:
            kk = 0
            for j in range(jend-radius, jend):
                j = j - jstart
                for i in range(istart, iend+1):
                    i = i - istart
                    top_buf_out[kk] = A[j+radius][i+radius]
                    kk += 1
            shmem.put(bottom_buf_in[sw], top_buf_out, top_nbr, radius * width[0])
            if splitfence:
                shmem.fence()
                shmem.atomic_inc(iterflag[sw:sw+1], top_nbr)

        if mey > 0:
            kk = 0
            for j in range(jstart, jstart+radius):
                j = j - jstart
                for i in range(istart, iend+1):
                    i = i - istart
                    bottom_buf_out[kk] = A[j+radius][i+radius]
                    kk += 1
            shmem.put(top_buf_in[sw], bottom_buf_out, bottom_nbr, radius*width[0])
            if splitfence:
                shmem.fence()
                shmem.atomic_inc(iterflag[sw:sw+1], bottom_nbr)

        if mex < npx-1:
            kk = 0
            for j in range(jstart, jend+1):
                j = j - jstart
                for i in range(iend-radius, iend):
                    i = i - istart
                    right_buf_out[kk] = A[j+radius][i+radius]
                    kk += 1
            shmem.put(left_buf_in[sw], right_buf_out, right_nbr, radius*height[0])
            if splitfence:
                shmem.fence()
                shmem.atomic_inc(iterflag[sw:sw+1], right_nbr)

        if mex > 0:
            kk = 0
            for j in range(jstart, jend+1):
                j = j - jstart
                for i in range(istart, istart+radius):
                    i = i - istart
                    left_buf_out[kk] = A[j+radius][i+radius]
                    kk += 1
            shmem.put(right_buf_in[sw], left_buf_out, left_nbr, radius*height[0])
            if splitfence:
                shmem.fence()
                shmem.atomic_inc(iterflag[sw:sw+1], left_nbr)

        if not splitfence:
            shmem.fence()

            if mey < npy-1:
                shmem.atomic_inc(iterflag[sw:sw+1], top_nbr)
            if mey > 0:
                shmem.atomic_inc(iterflag[sw:sw+1], bottom_nbr)
            if mex < npx-1:
                shmem.atomic_inc(iterflag[sw:sw+1], right_nbr)
            if mex > 0:
                shmem.atomic_inc(iterflag[sw:sw+1], left_nbr)

        shmem.wait_until(iterflag[sw:sw+1], shmem.CMP.EQ, count_case * (iter // 2 + 1))

        if mey < npy-1:
            kk = 0
            for j in range(jend, jend+radius):
                j = j - jstart
                for i in range(istart, iend+1):
                    i = i - istart
                    A[j+radius][i+radius] = top_buf_in[sw][kk]
                    kk += 1

        if mey > 0:
            kk = 0
            for j in range(jstart-radius, jstart):
                j = j - jstart
                for i in range(istart, iend+1):
                    i = i - istart
                    A[j+radius][i+radius] = bottom_buf_in[sw][kk]
                    kk += 1

        if mex < npx-1:
            kk = 0
            for j in range(jstart, jend+1):
                j = j - jstart
                for i in range(iend, iend+radius):
                    i = i - istart
                    A[j+radius][i+radius] = right_buf_in[sw][kk]
                    kk += 1

        if mex > 0:
            kk = 0
            for j in range(jstart, jend+1):
                j = j - jstart
                for i in range(istart-radius, istart):
                    i = i - istart
                    A[j+radius][i+radius] = left_buf_in[sw][kk]
                    kk += 1

        # Apply the stencil operator
        star(n,radius,A,B,weight,jstart,jend,istart,iend)

        # add constant to solution to force refresh of neighbor data, if any
        A[radius:jend-jstart+radius,radius:iend-istart+radius] += 1.0
        # numpy.add(A[radius:jend-jstart+radius,radius:iend-istart+radius], 1.0, A[radius:jend-jstart+radius,radius:iend-istart+radius])

    local_stencil_time[0] = time.monotonic() - local_stencil_time[0]
    shmem.barrier_all()
    shmem.max_reduce(stencil_time, local_stencil_time)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    local_norm[0] = 0.0
    for j in range(max(jstart, radius), min(n-radius, jend)):
        for i in range(max(istart, radius), min(n-radius, iend)):
            local_norm[0] += abs(B[j-jstart][i-istart])

    shmem.barrier_all()
    shmem.sum_reduce(norm, local_norm)

    # verify correctness
    active_points = (n-2*radius)**2
    if me == 0:
        epsilon = 1e-8
        norm[0] /= active_points
        if radius > 0:
            reference_norm = (iterations+1) * (2.0)
        else:
            reference_norm = 0.0
        if abs(norm[0]-reference_norm) > epsilon:
            print(f"ERROR: L1 norm = {norm[0]}, Reference L1 norm = {reference_norm}")
            shmem.global_exit(1)
        else:
            print(f"Reference L1 norm = {reference_norm}, L1 norm = {norm[0]}")

    if me == 0:
        # flops/stencil: 2 flops (fma) for each point in the stencil
        # plus one flop for the update of the input of the array
        stencil_size = 4*radius+1
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time[0]/iterations
        print(f"Rate (MFlops/s): {1.0E-06 * flops/avgtime}  Avg time (s): {avgtime}")


if __name__ == '__main__':
    main()
