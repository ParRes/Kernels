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
#include "global.fh"
#include "mafdecls.fh"
!#include 'ga-mpi.fh' ! unused
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
  integer :: A, B, AT
  integer :: mylo(2),myhi(2)
  real(kind=REAL64), parameter :: one  = 1.d0
  real(kind=REAL64), allocatable ::  T(:,:)
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) ::  order
  integer(kind=INT64) ::  bytes, max_mem
  ! runtime variables
  integer(kind=INT32) ::  i, j, k, ii, jj
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

  !call ga_initialize()
  ! ask GA to allocate enough memory for 4 matrices, just to be safe
  max_mem = order * order * 4 * ( storage_size(one) / 8 )
  call ga_initialize_ltd(max_mem)

  me = ga_nodeid()
  np = ga_nnodes()

  !if (me.eq.0) print*,'max_mem=',max_mem

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
    write(*,'(a47)') 'Fortran Global Arrays Matrix transpose: B = A^T'
    write(*,'(a22,i12)') 'Number of GA procs   = ', np
    write(*,'(a,i8)') 'Number of iterations    = ', iterations
    write(*,'(a,i8)') 'Matrix order            = ', order
  endif

  call ga_sync()

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  t0 = 0.0d0

  !print*,'order=',order
  ! must cast int32 order to integer...
  ok = ga_create(MT_DBL, int(order), int(order),'A',-1,-1, A)
  if (.not.ok) then
    call ga_error('allocation of A failed',100)
  endif

  ok = ga_duplicate(A,B,'B')
  if (.not.ok) then
    call ga_error('duplication of A as B failed ',101)
  endif
  call ga_zero(B)

  ok = ga_duplicate(A,AT,'A^T')
  if (.not.ok) then
    call ga_error('duplication of A as A^T failed ',102)
  endif
  call ga_zero(AT)

  call ga_sync()

  call ga_distribution( A, me, mylo(1), myhi(1), mylo(2), myhi(2) )
  !write(*,'(a7,5i6)') 'local:',me,mylo(1), myhi(1), mylo(2), myhi(2)
  allocate( T(myhi(1)-mylo(1)+1,myhi(2)-mylo(2)+1), stat=err)
  if (err .ne. 0) then
    call ga_error('allocation of T failed',err)
  endif
  do j=mylo(2),myhi(2)
    jj = j-mylo(2)+1
    do i=mylo(1),myhi(1)
      ii = i-mylo(1)+1
      T(ii,jj) = real(order,REAL64) * real(j-1,REAL64) + real(i-1,REAL64)
    enddo
  enddo
  !write(*,'(a8,5i6)') 'ga_put:',mylo(1), myhi(1), mylo(2), myhi(2), myhi(2)-mylo(2)+1
  call ga_put( A, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )
  call ga_sync()

  ok = ma_init(MT_DBL, order*order, 0)
  if (.not.ok) then
    call ga_error('ma_init failed', 1)
  endif

  if (order.lt.10) then
    call ga_print(A)
  endif

  do k=0,iterations

    ! start timer after a warmup iteration
    if (k.eq.1) then
        call ga_sync()
        t0 = ga_wtime()
    endif

    ! B += A^T
    ! A += 1
    call ga_transpose(A,AT)      ! C  = A^T
    call ga_sync()               ! ga_tranpose does not synchronize after remote updates
    call ga_add(one,B,one,AT,B)  ! B += A^T
    call ga_add_constant(A, one) ! A += 1
    !call ga_sync()               ! ga_add and ga_add_constant synchronize

  enddo ! iterations

  call ga_sync()
  t1 = ga_wtime()

  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  if (order.lt.10) then
    call ga_print(A)
    call ga_print(AT)
    call ga_print(B)
  endif

  !write(*,'(a8,5i6)') 'ga_get:',mylo(1), myhi(1), mylo(2), myhi(2), myhi(2)-mylo(2)+1
  call ga_get( B, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )

  abserr = 0.0
  ! this will overflow if iterations>>1000
  addit = (0.5*iterations) * (iterations+1)
  do j=mylo(2),myhi(2)
    jj = j-mylo(2)+1
    do i=mylo(1),myhi(1)
      ii = i-mylo(1)+1
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(j-1,REAL64)) &
           * real(iterations+1,REAL64)
      abserr = abserr + abs(T(ii,jj) - (temp+addit))
    enddo
  enddo
  call ga_dgop(MT_DBL, abserr, 1, '+')

  deallocate( T )

  ok = ga_destroy(AT)
  if (.not.ok) then
      call ga_error('ga_destroy failed',201)
  endif

  ok = ga_destroy(A)
  if (.not.ok) then
      call ga_error('ga_destroy failed',202)
  endif

  ok = ga_destroy(B)
  if (.not.ok) then
      call ga_error('ga_destroy failed',203)
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
  if (me.eq.0) write(*,'(a)') ! add an empty line
  call ga_summarize(.false.)
  if (me.eq.0) then
    call ma_print_stats()
  endif
#endif

  call ga_terminate()
  call mpi_finalize()

end program main

