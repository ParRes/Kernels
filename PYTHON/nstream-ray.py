#!/usr/bin/env python3
#
# Copyright (c) 2024, Intel Corporation
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
# NAME:    nstream
#
# PURPOSE: To compute memory bandwidth when adding a vector of a given
#          number of double precision values to the scalar multiple of
#          another vector of the same length, and storing the result in
#          a third vector.
#
# USAGE:   The program takes as input the number
#          of iterations to loop over the triad vectors, the length of the
#          vectors, and the offset between vectors
#
#          <progname> <# iterations> <vector length> <offset>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# NOTES:   Bandwidth is determined as the number of words read, plus the
#          number of words written, times the size of the words, divided
#          by the execution time. For a vector length of N, the total
#          number of words read and written is 4*N*sizeof(double).
#
#
# HISTORY: This code is loosely based on the Stream benchmark by John
#          McCalpin, but does not follow all the Stream rules. Hence,
#          reported results should not be associated with Stream in
#          external publications
#
#          Converted to Python by Jeff Hammond, October 2017.
#          Adapted to Ray by AI Assistant, 2024.
#
# *******************************************************************

import sys
import ray
import numpy
import time

@ray.remote
def nstream_chunk(A_chunk, B_chunk, C_chunk, iterations, scalar):
    """
    Ray remote function to compute STREAM triad on pre-initialized vector chunks
    A = B + scalar * C for given array chunks
    """
    # Ensure we have writable copies of the arrays
    A = A_chunk.copy()
    B = B_chunk.copy() 
    C = C_chunk.copy()
    
    # Perform the STREAM triad iterations on the provided chunks
    for k in range(iterations + 1):
        A += B + scalar * C
    
    # Return the final A chunk for validation
    return A

@ray.remote
def validate_chunk(A_chunk):
    """
    Ray remote function to compute validation sum for a chunk
    This is separated from computation to avoid timing validation
    """
    # Return the L1 norm of A for validation
    asum = numpy.linalg.norm(A_chunk, ord=1)
    return asum

def main():
    # Initialize Ray (start local cluster)
    ray.init(ignore_reinit_error=True)
    
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ')
    print('Python Ray/Numpy STREAM triad: A = B + scalar * C')
    print('Ray version = ', ray.__version__)
    print('Numpy version = ', numpy.version.version)

    if len(sys.argv) != 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: python nstream-ray.py <# iterations> <vector length>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    total_length = int(sys.argv[2])
    if total_length < 1:
        sys.exit("ERROR: length must be positive")

    # Get number of available cores/workers
    num_workers = ray.cluster_resources().get('CPU', 1)
    num_workers = int(num_workers)
    
    # Calculate chunk size - similar to MPI domain decomposition
    length_per_worker = int(total_length / num_workers)
    remainder = total_length % num_workers
    
    # Create chunk sizes (some workers may get one extra element)
    chunk_sizes = []
    for i in range(num_workers):
        chunk_size = length_per_worker
        if i < remainder:
            chunk_size += 1
        chunk_sizes.append(chunk_size)

    print('Number of workers    = ', num_workers)
    print('Number of iterations = ', iterations)
    print('Vector length        = ', total_length)
    print('Chunk sizes          = ', chunk_sizes)

    scalar = 3.0

    # ********************************************************************
    # ** Allocate and initialize arrays (not timed)
    # ********************************************************************
    
    # Initialize full arrays
    A = numpy.zeros(total_length)
    B = numpy.full(total_length, 2.0)
    C = numpy.full(total_length, 2.0)
    
    # Split arrays into chunks for distribution
    A_chunks = []
    B_chunks = []
    C_chunks = []
    start_idx = 0
    
    for chunk_size in chunk_sizes:
        if chunk_size > 0:
            end_idx = start_idx + chunk_size
            A_chunks.append(A[start_idx:end_idx].copy())
            B_chunks.append(B[start_idx:end_idx].copy())
            C_chunks.append(C[start_idx:end_idx].copy())
            start_idx = end_idx

    # ********************************************************************
    # ** Execute STREAM triad in parallel using Ray
    # ********************************************************************

    # Start timing
    t0 = time.time()
    
    # Launch parallel computation tasks with pre-initialized chunks
    compute_futures = []
    for i, chunk_size in enumerate(chunk_sizes):
        if chunk_size > 0:  # Only launch tasks for non-empty chunks
            future = nstream_chunk.remote(A_chunks[i], B_chunks[i], C_chunks[i], iterations, scalar)
            compute_futures.append(future)
    
    # Wait for all computation tasks to complete
    chunk_results = ray.get(compute_futures)
    
    t1 = time.time()
    nstream_time = t1 - t0

    # ********************************************************************
    # ** Compute validation (not timed)
    # ********************************************************************
    
    # Launch validation tasks
    validation_futures = []
    for A_chunk in chunk_results:
        future = validate_chunk.remote(A_chunk)
        validation_futures.append(future)
    
    # Wait for all validation tasks to complete
    chunk_asums = ray.get(validation_futures)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    # Calculate expected result
    ar = 0.0
    br = 2.0
    cr = 2.0
    for k in range(iterations + 1):
        ar += br + scalar * cr

    ar *= total_length

    # Sum up results from all chunks
    asum = sum(chunk_asums)

    epsilon = 1.e-8
    if abs(ar - asum) / asum > epsilon:
        print('Failed Validation on output array')
        print('        Expected checksum: ', ar)
        print('        Observed checksum: ', asum)
        sys.exit("ERROR: solution did not validate")
    else:
        print('Solution validates')
        avgtime = nstream_time / iterations
        nbytes = 4.0 * total_length * 8  # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
        print('Rate (MB/s): ', 1.e-6 * nbytes / avgtime, ' Avg time (s): ', avgtime)

    # Shutdown Ray
    ray.shutdown()

if __name__ == '__main__':
    main()
