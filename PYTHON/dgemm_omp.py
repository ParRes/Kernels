#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
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
# NAME:    dgemm
#
# PURPOSE: This program tests the efficiency with which a dense matrix
#          dense multiplication is carried out
#
# USAGE:   The program takes as input the matrix order,
#          the number of times the matrix-matrix multiplication 
#          is carried out.
#
#          <progname> <# iterations> <matrix order>
#
#          The output consists of diagnostics to make sure the 
#          algorithm worked, and of timing statistics.
#
# HISTORY: Written by Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#          PyOMP support, ave+std_dev by Tim Mattson, May 2021
# *******************************************************************

import sys
from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_set_num_threads, omp_get_thread_num, omp_get_num_threads, omp_get_wtime
import numpy as np
#from time import process_time as timer

#@njit(enable_ssa=False, cache=True)    What does "enable_ssa" mean? 
@njit(fastmath=True)
def dgemm(iters,order):
    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    print('inside dgemm')
    A = np.zeros((order,order))
    B = np.zeros((order,order))
    C = np.zeros((order,order))

    for i in range(order):
        A[:,i] = float(i)
        B[:,i] = float(i)

#    print(omp_get_num_threads())
    for kiter in range(0,iters+1):
         if kiter==1: 
             t0 = omp_get_wtime()
             tSum=0.0
             tsqSum=0.0
         with openmp("parallel for schedule(static) private(j,k)"):
               for i in range(order):
                   for k in range(order):
                       for j in range(order):
                           C[i][j] += A[i][k] * B[k][j]
         if kiter>0:
             tkiter = omp_get_wtime()
             t = tkiter - t0
             tSum = tSum + t
             tsqSum = tsqSum+t*t
             t0 = tkiter

    dgemmAve    = tSum/iters
    dgemmStdDev = ((tsqSum-iters*dgemmAve*dgemmAve)/(iters-1))**0.5 
    print('finished with computations')

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    checksum = 0.0;
    for i in range(order):
        for j in range(order):
            checksum += C[i][j];

    ref_checksum = order*order*order
    ref_checksum *= 0.25*(order-1.0)*(order-1.0)
    ref_checksum *= (iters+1)
    epsilon=1.e-8
    if abs((checksum - ref_checksum)/ref_checksum) < epsilon:
        print('Solution validates')
        nflops = 2.0*order*order*order
        recipDiff = (1.0/(dgemmAve-dgemmStdDev) - 1.0/(dgemmAve+dgemmStdDev))
        GfStdDev = 1.e-6*nflops*recipDiff/2.0
        print('nflops: ',nflops)
        print('Rate: ',1.e-6*nflops/dgemmAve,' +/- (MF/s): ',GfStdDev)
    else:
        print('ERROR: Checksum = ', checksum,', Reference checksum = ', ref_checksum,'\n')
#        sys.exit("ERROR: solution did not validate")


# ********************************************************************
# read and test input parameters
# ********************************************************************

print('Parallel Research Kernels version ') #, PRKVERSION
print('Python Dense matrix-matrix multiplication: C = A x B')

if len(sys.argv) != 3:
   print('argument count = ', len(sys.argv))
   sys.exit("Usage: ./dgemm <# iterations> <matrix order>")

itersIn = int(sys.argv[1])
if itersIn < 1:
   sys.exit("ERROR: iterations must be >= 1")

orderIn = int(sys.argv[2])
if orderIn < 1:
    sys.exit("ERROR: order must be >= 1")

print('Number of iterations = ', itersIn)
print('Matrix order         = ', orderIn)

dgemm(itersIn, orderIn)

