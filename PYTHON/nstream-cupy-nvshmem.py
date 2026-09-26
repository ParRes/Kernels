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
#          Adapted for CuPy+NVSHMEM4Py, December 2024.
#
# *******************************************************************

import sys
# A wall-clock timer, not time.process_time(): unlike the other PRK Python
# scripts (CPU-bound, where process time and wall time track closely), the
# STREAM triad below runs asynchronously on the GPU, and process time barely
# advances while the host blocks waiting on the device -- it would badly
# undercount the real elapsed time.
from timeit import default_timer as timer

from mpi4py import MPI

import cupy

if False:
    print('=== CUDA Version Information ===')

    try:
        # Get CUDA runtime version
        runtime_version = cupy.cuda.runtime.runtimeGetVersion()
        runtime_major = runtime_version // 1000
        runtime_minor = (runtime_version % 1000) // 10
        print(f'CUDA Runtime Version: {runtime_major}.{runtime_minor} (raw: {runtime_version})')
        
        # Get CUDA driver version  
        driver_version = cupy.cuda.runtime.driverGetVersion()
        driver_major = driver_version // 1000
        driver_minor = (driver_version % 1000) // 10
        print(f'CUDA Driver Version: {driver_major}.{driver_minor} (raw: {driver_version})')
        
        print(f'Version compatibility: Driver {driver_major}.{driver_minor} vs Runtime {runtime_major}.{runtime_minor}')
        
        if driver_version < runtime_version:
            print('WARNING: Driver version is older than runtime version!')
            print('This can cause \"cudaErrorInsufficientDriver\" errors.')
            print('Consider updating your NVIDIA drivers.')
        else:
            print('Driver and runtime versions are compatible.')
            
    except Exception as e:
        print(f'Error: {e}')
    print('This usually indicates CUDA driver/runtime compatibility issues.')

from cuda.core import Device
from cuda.core import system

import nvshmem.core as nvshmem

def main():

    # Initialize MPI and CUDA device
    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank() % system.get_num_devices()
    device = Device(local_rank)
    device.set_current()
    stream = device.create_stream()
    
    # Initialize NVSHMEM with MPI
    nvshmem.init(device=device, mpi_comm=comm, initializer_method="mpi")
    
    me = nvshmem.my_pe()
    np = nvshmem.n_pes()

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    if (me==0):
        print('Parallel Research Kernels version ') #, PRKVERSION
        print('Python CuPy/NVSHMEM STREAM triad: A = B + scalar * C')

    if len(sys.argv) != 3:
        if (me==0):
            print('argument count = ', len(sys.argv))
            print("Usage: python nstream-cupy-nvshmem.py <# iterations> <vector length>")
        nvshmem.finalize()
        sys.exit()

    iterations = int(sys.argv[1])
    if iterations < 1:
        if (me==0):
            print("ERROR: iterations must be >= 1")
        nvshmem.finalize()
        sys.exit()

    total_length = int(sys.argv[2])
    if total_length < 1:
        if (me==0):
            print("ERROR: length must be positive")
        nvshmem.finalize()
        sys.exit()

    # Distribute work across GPUs/PEs
    length = int(total_length / np)
    remainder = total_length % np
    if (remainder > 0):
        if (me < remainder):
            length += 1

    if (me==0):
        print('Number of PEs        = ', np)
        print('Number of iterations = ', iterations)
        print('Vector length        = ', total_length)
        print('Local vector length  = ', length)

    # Barrier using NVSHMEM
    nvshmem.barrier(nvshmem.Teams.TEAM_WORLD,stream=stream)
    stream.sync()

    # ********************************************************************
    # ** Allocate space for the input and execute STREAM triad
    # ********************************************************************

    # Allocate symmetric GPU arrays using NVSHMEM4Py interoperability with CuPy
    A = nvshmem.interop.cupy.array((length,), dtype="float64")
    B = nvshmem.interop.cupy.array((length,), dtype="float64") 
    C = nvshmem.interop.cupy.array((length,), dtype="float64")
    
    # Initialize arrays
    A[:] = 0.0
    B[:] = 2.0
    C[:] = 2.0

    scalar = 3.0

    # Timing loop
    # CuPy operations issued with no explicit stream run on CuPy's own
    # default stream, a different stream than the cuda.core `stream` used
    # for NVSHMEM barrier/sync above -- so stream.sync() alone was never
    # actually waiting for the triad kernels below to finish, and the
    # timing was measuring almost nothing. Bind CuPy's current stream to
    # the same one NVSHMEM uses so the two are properly ordered.
    with cupy.cuda.Stream.from_external(stream):
        for k in range(0, iterations+1):

            if k < 1:
                nvshmem.barrier(nvshmem.Teams.TEAM_WORLD,stream=stream)
                stream.sync()
                t0 = timer()

            # STREAM triad operation on GPU using CuPy operations
            A += B + scalar * C

    # Final synchronization
    nvshmem.barrier(nvshmem.Teams.TEAM_WORLD,stream=stream)
    stream.sync()
    t1 = timer()
    nstream_time = t1 - t0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    # Calculate expected result
    ar = 0.0
    br = 2.0
    cr = 2.0
    for k in range(0, iterations+1):
        ar += br + scalar * cr

    ar *= total_length

    # Calculate local checksum
    asum_local = cupy.linalg.norm(A, ord=1)
    
    # Create source and destination arrays for reduction
    src = nvshmem.interop.cupy.array((1,), dtype="float64")
    dst = nvshmem.interop.cupy.array((1,), dtype="float64")
    src[0] = asum_local
    dst[0] = 0.0
    
    # Reduce across all PEs using NVSHMEM collective
    nvshmem.reduce(nvshmem.Teams.TEAM_WORLD, dst, src, op="sum", stream=stream)
    stream.sync()
    
    asum_global = float(dst[0])

    epsilon = 1.e-8
    if abs(ar - asum_global) / asum_global > epsilon:
        if (me == 0):
            print('Failed Validation on output array')
            print('        Expected checksum: ', ar)
            print('        Observed checksum: ', asum_global)
            print("ERROR: solution did not validate")
    else:
        if (me == 0):
            print('Solution validates')
            avgtime = nstream_time / iterations
            nbytes = 4.0 * total_length * 8  # 8 bytes per double
            print('Rate (GB/s): ', 1.e-9 * nbytes / avgtime, ' Avg time (s): ', avgtime)

    # Free NVSHMEM arrays
    nvshmem.free_array(A)
    nvshmem.free_array(B)
    nvshmem.free_array(C)
    nvshmem.free_array(src)
    nvshmem.free_array(dst)
    
    # Finalize NVSHMEM
    nvshmem.finalize()


if __name__ == '__main__':
    main()
