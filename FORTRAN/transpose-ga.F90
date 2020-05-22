!
! Copyright (c) 2015, Intel Corporation
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
! NAME:    transpose
!
! PURPOSE: This program measures the time for the transpose of a
!          column-major stored matrix into a row-major stored matrix.
!
! USAGE:   Program input is the matrix order and the number of times to
!          repeat the operation:
!
!          transpose <matrix_size> <# iterations> [tile size]
!
!          An optional parameter specifies the tile size used to divide the
!          individual matrix blocks for improved cache and TLB performance.
!
!          The output consists of diagnostics to make sure the
!          transpose worked and timing statistics.
!
! HISTORY: Written by  Rob Van der Wijngaart, February 2009.
!          Converted to Fortran by Jeff Hammond, January 2015
!          Global Arrays by Jeff Hammond, May 2020.
! *******************************************************************

program main
  use iso_fortran_env
  use mpi_f08
  implicit none
#include 'global.fh'
!#include 'ga-mpi.fh' ! unused
#include 'mafdecls.fh'
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
  integer :: A, B, C
  real(kind=REAL64), parameter :: zero = 0.d0
  real(kind=REAL64), parameter :: one  = 1.d0
  real(kind=REAL64), parameter :: two  = 2.d0
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) ::  order
  integer(kind=INT64) ::  bytes, max_mem
  ! runtime variables
  integer(kind=INT32) ::  i, j, k
  real(kind=REAL64) ::  abserr, addit, temp
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.d-8

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a62)')    'Usage: ./transpose <# iterations> <matrix order>'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  order = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') order
  if (order .lt. 1) then
    write(*,'(a,i5)') 'ERROR: order must be >= 1 : ', order
    stop 1
  endif

  call mpi_init_thread(requested,provided)

  ! ask GA to allocate enough memory for 3 matrices, just to be safe
  max_mem = order * order * 3 * ( storage_size(zero) / 8 )
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
    write(*,'(a44)') 'Fortran Global Arrays Matrix transpose: B = A^T'
    write(*,'(a22,i12)') 'Number of GA procs   = ', np
    write(*,'(a,i8)') 'Number of iterations    = ', iterations
    write(*,'(a,i8)') 'Matrix order            = ', order
  endif

  call ga_sync()

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  t0 = 0.0d0

  ok = ga_create(MT_DBL, order, order,'A',-1,-1, A)
  if (.not.ok) then
    call ga_error('allocation of A failed',100)
  endif

  ok = ga_duplicate(A,B,'B')
  if (.not.ok) then
    call ga_error('duplication of A as B failed ',101)
  endif
  call ga_zero(B)
  call ga_sync()

  call ga_distribution(g_a, iproc, ilo, ihi, jlo, jhi)



    do j=1,order
      do i=1,order
!        A(i,j) = real(order,REAL64) * real(j-1,REAL64) + real(i-1,REAL64)
      enddo
    enddo


  do k=0,iterations

    ! start timer after a warmup iteration
    if (k.eq.1) then
        call ga_sync()
        t0 = ga_wtime()
    endif

    ! Transpose the  matrix; only use tiling if the tile size is smaller than the matrix
    do j=1,order
      do i=1,order
!        B(j,i) = B(j,i) + A(i,j)
!        A(i,j) = A(i,j) + one
      enddo
    enddo

  enddo ! iterations

  call ga_sync()
  t1 = ga_wtime()

  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  abserr = 0.0
  ! this will overflow if iterations>>1000
  addit = (0.5*iterations) * (iterations+1)
  do j=1,order
    do i=1,order
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(j-1,REAL64)) &
           * real(iterations+1,REAL64)
!      abserr = abserr + abs(B(i,j) - (temp+addit))
    enddo
  enddo

  ok = ga_destroy(A)
  if (.not.ok) then
      call ga_error('ga_destroy failed',201)
  endif

  ok = ga_destroy(B)
  if (.not.ok) then
      call ga_error('ga_destroy failed',202)
  endif

  call ga_sync()

  if (me.eq.0) then
    if (abserr .lt. epsilon) then
      write(*,'(a)') 'Solution validates'
      avgtime = trans_time/iterations
      bytes = 2 * int(order,INT64) * int(order,INT64) * storage_size(one)/8
      write(*,'(a,f13.6,a,f10.6)') 'Rate (MB/s): ',(1.d-6*bytes)/avgtime, &
             ' Avg time (s): ', avgtime
    else
      write(*,'(a,f30.15,a,f30.15)') 'ERROR: Aggregate squared error ',abserr, &
             'exceeds threshold ',epsilon
      call ga_error('Answer wrong',911)
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

