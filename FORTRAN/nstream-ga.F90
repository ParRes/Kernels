!
! Copyright (c) 2017, Intel Corporation
! Copyright (c) 2021, NVIDIA
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
! * Redistributions of source code must retain the above copyright
!      notice, this list of conditions and the following disclaimer.
! * Redistributions in binary form must reproduce the above
!      copyright notice, this list of conditions and the following
!      disclaimer in the documentation and/or other materials provided
!      with the distribution.
! * Neither the name of Intel Corporation nor the names of its
!      contributors may be used to endorse or promote products
!      derived from this software without specific prior written
!      permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.

!*******************************************************************
!
! NAME:    nstream
!
! PURPOSE: To compute memory bandwidth when adding a vector of a given
!          number of double precision values to the scalar multiple of
!          another vector of the same length, and storing the result in
!          a third vector.
!
! USAGE:   The program takes as input the number
!          of iterations to loop over the triad vectors, the length of the
!          vectors, and the offset between vectors
!
!          <progname> <# iterations> <vector length> <offset>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! NOTES:   Bandwidth is determined as the number of words read, plus the
!          number of words written, times the size of the words, divided
!          by the execution time. For a vector length of N, the total
!          number of words read and written is 4*N*sizeof(double).
!
!
! HISTORY: This code is loosely based on the Stream benchmark by John
!          McCalpin, but does not follow all the Stream rules. Hence,
!          reported results should not be associated with Stream in
!          external publications
!
!          Converted to C++11 by Jeff Hammond, 2017.
!          Converted to Fortran by Jeff Hammond, 2017.
!          Global Arrays by Jeff Hammond, May 2020.
!
! *******************************************************************

program main
  use iso_fortran_env
  use mpi_f08
  implicit none
#include "global.fh"
!#include 'ga-mpi.fh' ! unused
#include "mafdecls.fh"
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! MPI - should always use 32-bit INTEGER
  integer(kind=INT32), parameter :: requested = MPI_THREAD_SERIALIZED
  integer(kind=INT32) :: provided
  integer(kind=INT32) :: world_size, world_rank
  integer(kind=INT32) :: ierr
  type(MPI_Comm), parameter :: world = MPI_COMM_WORLD
  ! GA - compiled with 64-bit INTEGER
  logical :: ok
  integer :: me, np
  integer, parameter :: ndim = 1
  integer :: dims(ndim)
  integer :: chunk(ndim)
  integer :: A, B, C
  real(kind=REAL64), parameter :: zero = 0.d0
  real(kind=REAL64), parameter :: one  = 1.d0
  real(kind=REAL64), parameter :: two  = 2.d0
  ! problem definition
  integer(kind=INT32) ::  iterations, offset
  integer(kind=INT64) ::  length
  integer(kind=INT64) :: bytes, max_mem
  real(kind=REAL64) :: scalar
  ! runtime variables
  integer(kind=INT64) :: i
  integer(kind=INT32) :: k
  real(kind=REAL64) ::  asum, ar, br, cr, atmp
  real(kind=REAL64) ::  t0, t1, nstream_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.d-8

  if (storage_size(length).ne.storage_size(me)) then
    write(*,'(a50)') 'You must compile with 64-bit INTEGER!'
    stop 1
  endif
  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a62)')    'Usage: ./transpose <# iterations> <vector length> [<offset>]'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  length = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') length
  if (length .lt. 1) then
    write(*,'(a,i5)') 'ERROR: length must be nonnegative : ', length
    stop 1
  endif

  offset = 0
  if (command_argument_count().gt.2) then
    call get_command_argument(3,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') offset
    if (offset .lt. 0) then
      write(*,'(a,i5)') 'ERROR: offset must be positive : ', offset
      stop 1
    endif
  endif

  call mpi_init_thread(requested,provided)

  ! ask GA to allocate enough memory for 4 vectors, just to be safe
  max_mem = length * 4 * ( storage_size(scalar) / 8 )
  call ga_initialize_ltd(max_mem)

  me = ga_nodeid()
  np = ga_nnodes()

#if PRK_CHECK_GA_MPI
  ! We do use MPI anywhere, but if we did, we would need to avoid MPI collectives
  ! on the world communicator, because it is possible for that to be larger than
  ! the GA world process group.  In this case, we need to get the MPI communicator
  ! associated with GA world, but those routines assume MPI communicators are integers.

  call MPI_Comm_rank(world, world_rank)
  call MPI_Comm_size(world, world_size)

  if ((me.ne.world_rank).or.(np.ne.world_size)) then
      write(*,'(a12,i8,i8)') 'rank=',me,world_rank
      write(*,'(a12,i8,i8)') 'size=',me,world_size
      call ga_error('MPI_COMM_WORLD is unsafe to use!!!',np)
  endif
#endif

  if (me.eq.0) then
    write(*,'(a25)') 'Parallel Research Kernels'
    write(*,'(a54)') 'Fortran Global Arrays STREAM triad: A = B + scalar * C'
    write(*,'(a22,i12)') 'Number of GA procs   = ', np
    write(*,'(a22,i12)') 'Number of iterations = ', iterations
    write(*,'(a22,i12)') 'Vector length        = ', length
    write(*,'(a22,i12)') 'Offset               = ', offset
  endif

  call ga_sync()

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  t0 = 0.0d0

  scalar = 3.0d0

  dims(1)  = length
  chunk(1) = -1

  ok = nga_create(MT_DBL, ndim, dims,'A', chunk, A)
  if (.not.ok) then
    call ga_error('allocation of A failed',100)
  endif

  ok = ga_duplicate(A,B,'B')
  if (.not.ok) then
    call ga_error('duplication of A as B failed ',101)
  endif

  ok = ga_duplicate(B,C,'C')
  if (.not.ok) then
    call ga_error('duplication of B as C failed ',101)
  endif

  call ga_sync()

  call ga_fill(A,zero)
  call ga_fill(B,two)
  call ga_fill(C,two)

  call ga_sync()

  do k=0,iterations

    ! start timer after a warmup iteration
    if (k.eq.1) then
        call ga_sync()
        t0 = ga_wtime()
    endif

    ! A += B + scalar * C
    call ga_add(one,A,one,B,A)     ! A = A + B
    call ga_add(one,A,scalar,C,A)  ! A = A + scalar * C

  enddo ! iterations

  call ga_sync()
  t1 = ga_wtime()

  nstream_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ar  = zero
  br  = two
  cr  = two
  do k=0,iterations
      ar = ar + br + scalar * cr;
  enddo

  call ga_add_constant(A,-ar)
  call ga_norm1(A,asum)
  call ga_sync()

  ok = ga_destroy(A)
  if (.not.ok) then
      call ga_error('ga_destroy failed',201)
  endif

  ok = ga_destroy(B)
  if (.not.ok) then
      call ga_error('ga_destroy failed',202)
  endif

  ok = ga_destroy(C)
  if (.not.ok) then
      call ga_error('ga_destroy failed',203)
  endif

  call ga_sync()

  if (me.eq.0) then
    if (abs(asum) .gt. epsilon) then
      write(*,'(a35)') 'Failed Validation on output array'
      write(*,'(a30,f30.15)') '       Expected value: ', ar
      !write(*,'(a30,f30.15)') '       Observed value: ', A(1)
      write(*,'(a35)')  'ERROR: solution did not validate'
      stop 1
      call ga_error('Answer wrong',911)
    else
      write(*,'(a17)') 'Solution validates'
      avgtime = nstream_time/iterations;
      bytes = 4 * int(length,INT64) * storage_size(scalar)/8
      write(*,'(a12,f15.3,1x,a12,e15.6)')    &
              'Rate (MB/s): ', 1.d-6*bytes/avgtime, &
              'Avg time (s): ', avgtime
    endif
  endif

  call ga_sync()

#ifdef PRK_GA_SUMMARY
  if (me.eq.0) then
    write(*,'(a)') ! add an empty line
  endif
  call ga_summarize(.false.)
#endif

  call ga_terminate()
  call mpi_finalize()

end program main

