!
! Copyright (c) 2015, Intel Corporation
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
!          MPI by Jeff Hammond, November 2021
! *******************************************************************

program main
  use iso_fortran_env
  use mpi_f08
  implicit none
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) ::  order
  real(kind=REAL64), allocatable ::  A(:,:)         ! buffer to hold original matrix
  real(kind=REAL64), allocatable ::  B(:,:)         ! buffer to hold transposed matrix
  real(kind=REAL64), allocatable ::  T(:,:)         ! temporary to hold tile
  real(kind=REAL64), parameter :: one=1.0d0
  integer(kind=INT64) ::  bytes
  ! distributed data helpers
  integer(kind=INT32) :: col_per_pe                 ! columns per PE = order/np
  integer(kind=INT32) :: col_start, row_start
  ! runtime variables
  integer(kind=INT32) ::  i, j, k, p, q
  integer(kind=INT32) ::  it, jt, tile_size
  real(kind=REAL64) ::  abserr, addit, temp
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.d-8
  ! MPI stuff
  integer(kind=INT32) :: me, np, provided

  call MPI_Init_thread(MPI_THREAD_FUNNELED,provided)
#ifdef _OPENMP
  if (provided.eq.MPI_THREAD_SINGLE) then
     call MPI_Abort(MPI_COMM_WORLD,1)
  endif
#endif
  call MPI_Comm_rank(MPI_COMM_WORLD, me)
  call MPI_Comm_size(MPI_COMM_WORLD, np)

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (me.eq.0) then
    write(*,'(a25)') 'Parallel Research Kernels'
#ifdef _OPENMP
    write(*,'(a43)') 'Fortran MPI/OpenMP Matrix transpose: B = A^T'
#else
    write(*,'(a36)') 'Fortran MPI Matrix transpose: B = A^T'
#endif

    if (command_argument_count().lt.2) then
      write(*,'(a17,i1)') 'argument count = ', command_argument_count()
      write(*,'(a62)')    'Usage: ./transpose <# iterations> <matrix order>'
      call MPI_Abort(MPI_COMM_WORLD, command_argument_count())
    endif
 
    iterations = 1
    call get_command_argument(1,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') iterations
    if (iterations .lt. 1) then
      write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
      call MPI_Abort(MPI_COMM_WORLD, 2)
    endif
 
    order = 1
    call get_command_argument(2,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') order
    if (order .lt. 1) then
      write(*,'(a,i5)') 'ERROR: order must be >= 1 : ', order
      call MPI_Abort(MPI_COMM_WORLD, 3)
    endif

#ifdef _OPENMP
    write(*,'(a23,i8)') 'Number of threads    = ', omp_get_max_threads()
#endif
    write(*,'(a23,i8)') 'Number of MPI procs  = ', np
    write(*,'(a23,i8)') 'Number of iterations = ', iterations
    write(*,'(a23,i8)') 'Matrix order         = ', order
  endif
  call MPI_Bcast(iterations, 1, MPI_INTEGER4, 0, MPI_COMM_WORLD)
  call MPI_Bcast(order, 1, MPI_INTEGER4, 0, MPI_COMM_WORLD)

  call MPI_Barrier(MPI_COMM_WORLD)

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  t0 = 0.0d0

  do k=0,iterations

    if (k.eq.1) then
        call MPI_Barrier(MPI_COMM_WORLD)
        t0 = MPI_Wtime()
    endif

    ! B += A^T
    ! A += 1

  enddo ! iterations

  call MPI_Barrier(MPI_COMM_WORLD)
  t1 = MPI_Wtime()

  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

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
      call MPI_Abort(MPI_COMM_WORLD,1)
    endif
  endif

  call MPI_Barrier(MPI_COMM_WORLD)
  call mpi_finalize()

end program main

