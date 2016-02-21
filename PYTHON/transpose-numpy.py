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
# *******************************************************************

import sys
import time
import numpy

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print 'Parallel Research Kernels version ' #, PRKVERSION
    print 'Python Matrix transpose: B = A^T'

    if len(sys.argv) != 3:
        print 'argument count = ', len(sys.argv)
        sys.exit("Usage: ./transpose <# iterations> <matrix n>")

    # iterations
    niter = int(sys.argv[1])
    if niter < 1: sys.exit("ERROR: niter must be >= 1")

    # matrix order
    n = int(sys.argv[2])
    if n < 1: sys.exit("ERROR: n must be >= 1")

    print 'Matrix n         = ', n
    print 'Number of niter = ', niter

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    A = numpy.fromfunction(lambda i,j: i*n+j, (n,n), dtype=float)
    B = numpy.zeros((n,n))

    for k in range(0,niter+1):
        # start timer after a warmup niteration
        if k<1: t0 = time.clock()

        # this actually forms the transpose of A
        # B += numpy.transpose(A)
        # this only uses the transpose _view_ of A
        B += A.T
        A += 1.0


    t1 = time.clock()
    trans_time = t1 - t0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    C = numpy.fromfunction(lambda i,j: ((niter/2.0)+(n*j+i))*(niter+1.0), (n,n), dtype=float)
    abserr = numpy.linalg.norm(numpy.reshape(B-C,n*n),ord=1)

    epsilon=1.e-8
    nbytes = 2 * n**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if abserr < epsilon:
        print 'Solution validates'
        avgtime = trans_time/niter
        print 'Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime
    else:
        print 'error ',abserr, ' exceeds threshold ',epsilon
        sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()

