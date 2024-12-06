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
  use, intrinsic :: iso_fortran_env
  use mpi_f08
  use prk
  use prk_mpi
  implicit none
  ! for argument parsing
  integer :: err
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) ::  order, block_order, bo2
  real(kind=REAL64), allocatable ::  A(:,:)         ! buffer to hold original matrix
  real(kind=REAL64), allocatable ::  B(:,:)         ! buffer to hold transposed matrix
  real(kind=REAL64), allocatable ::  T(:,:)         ! temporary to hold tile
  real(kind=REAL64), parameter :: one=1.0d0
  ! runtime variables
  integer(kind=INT64) :: bytes
  integer(kind=INT32) :: i, j, k, lo, hi, q, tile_size
  real(kind=REAL64) ::  abserr, addit, temp
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.d-8
  ! MPI stuff
  integer(kind=INT32) :: me, np, provided
  integer(kind=INT32) :: send_to, recv_from

  call MPI_Init_thread(MPI_THREAD_SINGLE,provided)
  call MPI_Comm_rank(MPI_COMM_WORLD, me)
  call MPI_Comm_size(MPI_COMM_WORLD, np)

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (me.eq.0) then
    call prk_get_arguments('transpose',iterations=iterations,order=order,tile_size=tile_size)
    write(*,'(a25)') 'Parallel Research Kernels'
    write(*,'(a37)') 'Fortran MPI Matrix transpose: B = A^T'
    write(*,'(a22,i8)') 'Number of MPI procs    = ', np
    write(*,'(a22,i8)') 'Number of iterations    = ', iterations
    write(*,'(a22,i8)') 'Matrix order            = ', order
    if (mod(order,np).ne.0) then
      write(*,'(a,2i5)') 'ERROR: order must an integer multiple of np : ', order,np
      call MPI_Abort(MPI_COMM_WORLD, 4)
    endif
  endif
  call MPI_Bcast(iterations, 1, MPI_INTEGER4, 0, MPI_COMM_WORLD)
  call MPI_Bcast(order, 1, MPI_INTEGER4, 0, MPI_COMM_WORLD)

  block_order = int(order / np)
  bo2 = block_order * block_order

  call MPI_Barrier(MPI_COMM_WORLD)

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(block_order,order), B(block_order,order), T(block_order,block_order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation  returned ',err
    stop 1
  endif
  
  ! Fill the original matrix
  do concurrent (i=1:order, j=1:block_order)
    A(j,i) = me * block_order + (i-1)*order + (j-1)
  end do
  B = 0

  t0 = 0.0d0

  do k=0,iterations

    if (k.eq.1) then
        call MPI_Barrier(MPI_COMM_WORLD)
        t0 = MPI_Wtime()
    endif

    ! B += A^T
    do q=0,np-1
        recv_from = mod( (me + q     ), np)
        send_to   = mod( (me - q + np), np)

        lo = block_order * send_to + 1
        hi = block_order * (send_to+1)
        call MPI_Sendrecv(A(:,lo:hi), bo2, MPI_DOUBLE_PRECISION, send_to, q, &
                          T, bo2, MPI_DOUBLE_PRECISION, recv_from, q,        &
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE)

        lo = block_order * recv_from + 1
        hi = block_order * (recv_from+1)
        B(:,lo:hi) = B(:,lo:hi) + transpose(T)
    end do
    ! A += 1
    A = A + one

  enddo ! iterations

  call MPI_Barrier(MPI_COMM_WORLD)
  t1 = MPI_Wtime()

  trans_time = t1 - t0

  deallocate( A,T )

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  abserr = 0.0;
  addit = (0.5*iterations) * (iterations+1.0)
  do j=1,block_order
    do i=1,order
      temp =  (order*(me*block_order+j-1)+(i-1)) * (iterations+1)+addit
      abserr = abserr + abs(B(j,i) - temp)
    enddo
  enddo
  call MPI_Allreduce(MPI_IN_PLACE,abserr,1,MPI_DOUBLE_PRECISION, &
                     MPI_SUM,MPI_COMM_WORLD)

  deallocate( B )

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
      !call MPI_Abort(MPI_COMM_WORLD,1)
    endif
  endif

  call MPI_Finalize()

end program main

