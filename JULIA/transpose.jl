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
#          Converted to Julia by Jeff Hammond, June 2016.
# *******************************************************************

# ********************************************************************
# read and test input parameters
# ********************************************************************

println("Parallel Research Kernels version ") #, PRKVERSION)
println("Julia Matrix transpose: B = A^T")

if length(ARGS) != 2
    println("argument count = ", length(ARGS))
    println("Usage: ./transpose <# iterations> <matrix order>")
    exit(1)
end

argv = map(x->parse(Int64,x),ARGS)

# iterations
iterations = argv[1]
if iterations < 1
    println("ERROR: iterations must be >= 1")
    exit(2)
end

# matrix order
order = argv[2]
if order < 1
    println("ERROR: order must be >= 1")
    exit(3)
end

println("Order                    = ", order)
println("Number of iterations     = ", iterations)

# ********************************************************************
# ** Allocate space for the input and transpose matrix
# ********************************************************************

A = zeros(Float64,order,order)
B = zeros(Float64,order,order)
# Fill the original matrix
for i in 1:order
    for j in 1:order
        A[i,j] = order * (j-1) + (i-1)
    end
end

t0 = time_ns()

for k in 1:iterations+1
    # start timer after a warmup iteration
    if k==1
        t0 = time_ns()
    end

    for j in 1:order
        for i in 1:order
            B[i,j] += A[j,i]
            A[j,i] += 1.0
        end
    end
end

t1 = time_ns()
trans_time = (t1 - t0) * 1.e-9

# ********************************************************************
# ** Analyze and output results.
# ********************************************************************

addit = (0.5*iterations) * (iterations+1)
abserr = 0.0
for i in 1:order
  for j in 1:order
    temp = (order * (i-1) + (j-1)) * (iterations+1)
    abserr = abserr + abs(B[i,j] - (temp+addit))
  end
end

epsilon=1.e-8
nbytes = 2 * order^2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
if abserr < epsilon
    println("Solution validates")
    avgtime = trans_time/iterations
    println("Rate (MB/s): ",1.e-6*nbytes/avgtime, " Avg time (s): ", avgtime)
else
    println("error ",abserr, " exceeds threshold ",epsilon)
    println("ERROR: solution did not validate")
    exit(1)
end

