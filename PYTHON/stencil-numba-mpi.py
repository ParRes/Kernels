#!/usr/bin/env python3
#
# Copyright (c) 2020, Yijian Hu
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
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, February 2016.
#
# *******************************************************************

import sys
from mpi4py import MPI
import numpy
from numba import jit

@jit(nopython=True)
def star(n, r, A, B, W, jstart, jend, istart, iend):
    for a in range(max(jstart, r), min(n-r-1, jend)+1):
        a = a - jstart
        for b in range(max(istart, r), min(n-r-1, iend)+1):
            b = b - istart
            for k in range(2 * r + 1):
                B[a][b] += W[r][k] * A[a + k][b + r]
                B[a][b] += W[k][r] * A[a + r][b + k]

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = r/fac1
            break
    return fac1, fac2

def main():

    comm = MPI.COMM_WORLD
    me = comm.Get_rank() #My ID
    np = comm.Get_size() #Number of processor, NOT numpy
    x, y = factor(np)
    comm = comm.Create_cart([x,y])
    coords = comm.Get_coords(me)
    X = coords[0]
    Y = coords[1]

    x = int(x)
    y = int(y)

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    if me==0:
        print('Parallel Research Kernels ')
        print('Python MPI/Numba  Stencil execution on 2D grid')

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./stencil <# iterations> <array dimension> [<star/stencil> <radius>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    nsquare = n * n
    if nsquare < np:
        sys.exit(f"ERROR: grid size {nsquare} must be at least # ranks: {np}")


    if len(sys.argv) > 3:
        pattern = sys.argv[3]
    else:
        pattern = 'star'

    if len(sys.argv) > 4:
        r = int(sys.argv[4])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
        if (2*r+1) > n:
            sys.exit("ERROR: Stencil radius exceeds grid size")
    else:
        r = 2


    if me == 0:
        print('Number of ranks      = ', np)
        print('Number of iterations = ', iterations)
        print('Grid size            = ', n)
        if pattern == 'star':
            print('Type of stencil      = star')
        else:
            print('Type of stencil      = stencil')
        print('Radius of stencil    = ', r)
        print('Data type            = float 64 (double precision in C)')
        print('Compact representation of stencil loop body')



    W = numpy.zeros((2*r+1,2*r+1))
    if pattern == 'star':
        stencil_size = 4*r+1
        #vh = numpy.fromfunction(lambda i: 1./(2*r*(i-r)), (2*r+1,), dtype=float)
        #vh[r] = 0.0
        #W[:,r] = vh
        #W[r,:] = vh
        for i in range(1,r+1):
            W[r,r+i] = +1./(2*i*r)
            W[r+i,r] = +1./(2*i*r)
            W[r,r-i] = -1./(2*i*r)
            W[r-i,r] = -1./(2*i*r)

    else:
        stencil_size = (2*r+1)**2
        #W = numpy.fromfunction(lambda i,j: 1./(4 * numpy.maximum(numpy.abs(i-r),numpy.abs(j-r)) * (2*numpy.maximum(numpy.abs(i-r),numpy.abs(j-r)) - 1) * r),(2*r+1,2*r+1),dtype=float)
        #sign = numpy.fromfunction(lambda i,j: j-i,(2*r+1,2*r+1) )
        #sign = numpy.sign(sign[::-1])
        #temp = numpy.fromfunction(lambda x: 1./((x-r)*4*r),(2*r+1,),dtype=float)  #main diagonal
        #temp[r]=0
        #W = numpy.fill_diagonal(sign*W,temp)
        for j in range(1,r+1):
            for i in range(-j+1,j):
                W[r+i,r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i,r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j,r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j,r+i] = -1./(4*j*(2*j-1)*r)

            W[r+j,r+j]    = +1./(4*j*r)
            W[r-j,r-j]    = -1./(4*j*r)

    width = n//x
    leftover = n%x

    if X<leftover:
        istart = (width+1) * X
        iend = istart + width

    else:
        istart = (width+1) * leftover + width * (X-leftover)
        iend = istart + width - 1

    width = iend - istart + 1
    if width == 0 :
        sys.exit(f"ERROR: rank {me} has no work to do")

    height = n//y
    leftover = n%y
    if Y<leftover:
        jstart = (height+1) * Y
        jend = jstart + height

    else:
        jstart = (height+1) * leftover + height * (Y-leftover)
        jend = jstart + height - 1

    height = jend - jstart + 1
    if height == 0:
        sys.exit(f"ERROR: rank {me} has no work to do")

    if width < r or height < r:
        sys.exit(f"ERROR: rank {me} has work tile smaller then stencil radius")

    A = numpy.zeros((height+2*r,width+2*r))
    a = numpy.fromfunction(lambda i,j: i+istart+j+jstart,(height,width),dtype='d')
    A[r:-r,r:-r] = a
    B = numpy.zeros((height,width))
    typ = MPI.DOUBLE_PRECISION

    if Y < y-1:
        top_nbr   = comm.Get_cart_rank([X,Y+1])
    if Y > 0:
        bot_nbr   = comm.Get_cart_rank([X,Y-1])
    if X > 0:
        left_nbr  = comm.Get_cart_rank([X-1,Y])
    if X < x-1:
        right_nbr = comm.Get_cart_rank([X+1,Y])

    if np > 1:
        top_buf_in  = numpy.zeros(r*width)
        top_buf_out = numpy.zeros(r*width)
        bot_buf_in  = numpy.zeros(r*width)
        bot_buf_out = numpy.zeros(r*width)

        right_buf_in  = numpy.zeros(r*height)
        right_buf_out = numpy.zeros(r*height)
        left_buf_in  = numpy.zeros(r*height)
        left_buf_out = numpy.zeros(r*height)

    for i in range(0,iterations+1):
        if i<1:
            comm.Barrier()
            t0 = MPI.Wtime()

        if Y < y-1 :
            #req0 = comm.Irecv([top_buf_in, r*width, typ], source =top_nbr , tag =101 )
            # ^ I do not know why this does not work
            req0 = comm.Irecv(top_buf_in, source =top_nbr , tag =101 )
            kk=0
            for a in range(jend-r+1, jend+1):
                a = a - jstart
                for b in range(istart, iend+1):
                    b = b-istart
                    top_buf_out[kk] = A[a+r][b+r]
                    kk = kk+1
            req1 = comm.Isend(top_buf_out, dest =top_nbr, tag =99)

        if Y > 0 :
            req2 = comm.Irecv(bot_buf_in, source =bot_nbr , tag =99 )
            kk=0
            for a in range(jstart, jstart+r):
                a = a - jstart
                for b in range(istart, iend+1):
                    b = b-istart
                    bot_buf_out[kk] = A[a+r][b+r]
                    kk = kk+1
            req3 = comm.Isend(bot_buf_out, dest =bot_nbr, tag =101)

        if X < x-1 :
            req4 = comm.Irecv(right_buf_in, source =right_nbr , tag =1010)
            kk=0
            for a in range(jstart, jend+1):
                a = a - jstart
                for b in range(iend-r+1, iend+1):
                    b = b-istart
                    right_buf_out[kk] = A[a+r][b+r]
                    kk = kk+1
            req5 = comm.Isend(right_buf_out, dest =right_nbr, tag =990)

        if X > 0 :
            req6 = comm.Irecv(left_buf_in, source =left_nbr , tag =990 )
            kk=0
            for a in range(jstart, jend+1):
                a = a - jstart
                for b in range(istart, istart+r):
                    b = b-istart
                    left_buf_out[kk] = A[a+r][b+r]
                    kk = kk+1
            req7 = comm.Isend(left_buf_out, dest =left_nbr, tag =1010)


        if Y < y-1 :
            req0.wait()
            req1.wait()
            kk=0
            for a in range(jend+1, jend+r+1):
                a = a - jstart
                for b in range(istart, iend+1):
                    b = b-istart
                    A[a+r][b+r] = top_buf_in[kk]
                    kk = kk+1

        if Y > 0 :
            req2.wait()
            req3.wait()
            kk=0
            for a in range(jstart-r, jstart):
                a = a-jstart
                for b in range(istart, iend+1):
                    b = b-istart
                    A[a+r][b+r] = bot_buf_in[kk]
                    kk = kk+1

        if X > 0 :
            req6.wait()
            req7.wait()
            kk=0
            for a in range(jstart, jend+1):
                a = a - jstart
                for b in range(istart-r, istart):
                    b = b-istart
                    A[a+r][b+r] = left_buf_in[kk]
                    kk = kk+1

        if X < x-1 :
            req4.wait()
            req5.wait()
            kk=0
            for a in range(jstart, jend+1):
                a = a - jstart
                for b in range(iend+1, iend+r+1):
                    b = b-istart
                    A[a+r][b+r] = right_buf_in[kk]
                    kk = kk+1

        # Apply the stencil operator
        star(n,r,A,B,W,jstart,jend,istart,iend)
        A[r:jend-jstart+r+1,r:iend-istart+r+1] += 1.0
        # numpy.add(A[r:jend-jstart+r+1,r:iend-istart+r+1],1.0,A[r:jend-jstart+r+1,r:iend-istart+r+1]]

    local_time = numpy.array(MPI.Wtime() - t0 , dtype ='d')
    total_time = numpy.array(0 , dtype ='d')

    comm.Reduce([local_time , 1 , typ],[total_time , 1 , typ], op=MPI.MAX , root =0)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    # compute L1 norm in parallel
    local_norm = 0.0
    for a in range(max(jstart,r), min(n-r-1,jend)+1):
        for b in range(max(istart,r), min(n-r-1,iend)+1):
            local_norm = local_norm + abs(B[a-jstart][b-istart])

    local_norm = numpy.array(local_norm, dtype ='d')
    norm = numpy.array(0 , dtype ='d')
    comm.Reduce([local_norm, 1 , typ], [norm, 1, typ], op=MPI.SUM , root =0)

    if me == 0:
        epsilon=1.e-8
        active_points = (n-2*r)**2
        norm = norm / active_points
        if r > 0:
            ref_norm = (iterations+1)*(2.0)
        else:
            ref_norm = 0.0
        if abs(norm-ref_norm) < epsilon:
            print('Solution validates')
            flops = (2*stencil_size+1) * active_points
            avgtime = total_time/iterations
            print('Rate (MFlops/s): ',1.e-6*flops/avgtime, ' Avg time (s): ',avgtime)
        else:
            print('ERROR: L1 norm = ', norm,' Reference L1 norm = ', ref_norm)
            sys.exit()


if __name__ == '__main__':
    main()
