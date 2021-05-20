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
!  NAME:    dgemm
!
!  PURPOSE: This program tests the efficiency with which a dense matrix
!           dense multiplication is carried out
!
!  USAGE:   The program takes as input the matrix order and
!           the number of times the matrix-matrix multiplication
!           is carried out.
!
!           <progname> <# iterations> <matrix order>
!
!           The output consists of diagnostics to make sure the
!           algorithm worked, and of timing statistics.
!
!  HISTORY: Written by Rob Van der Wijngaart, February 2009.
!           Converted to C++11 by Jeff Hammond, December, 2017.
!           Converted to Fortran by Jeff Hammond, December, 2017.
!           Global Arrays by Jeff Hammond, May 2020.
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
  integer :: A, B, C
  integer :: mylo(2),myhi(2)
  real(kind=REAL64), parameter :: one  = 1.d0
  real(kind=REAL64), allocatable ::  T(:,:)
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) ::  order
  real(kind=REAL64)   :: forder
  integer(kind=INT64) ::  bytes, max_mem
  integer(kind=INT64) :: nflops
  ! runtime variables
  integer(kind=INT32) ::  i, j, k, ii, jj
  real(kind=REAL64) ::  checksum, reference, residuum
  real(kind=REAL64) ::  t0, t1, dgemm_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.d-8

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a62)')    'Usage: ./dgemm-ga <# iterations> <matrix order>'
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
    write(*,'(a68)') 'Fortran Global Arrays Dense matrix-matrix multiplication: C += A x B'
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
  call ga_zero(A)

  ok = ga_duplicate(A,B,'B')
  if (.not.ok) then
    call ga_error('duplication of A as B failed ',101)
  endif
  call ga_zero(B)

  ok = ga_duplicate(A,C,'C')
  if (.not.ok) then
    call ga_error('duplication of A as C failed ',102)
  endif
  call ga_zero(C)

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
      T(ii,jj) = i-1
    enddo
  enddo
  !write(*,'(a8,5i6)') 'ga_put:',mylo(1), myhi(1), mylo(2), myhi(2), myhi(2)-mylo(2)+1
  call ga_put( A, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )
  call ga_put( B, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )
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

    ! C = C + matmul(A,B)
    call ga_dgemm('n', 'n', int(order), int(order), int(order), one, A, B, one, C)

  enddo ! iterations

  call ga_sync()
  t1 = ga_wtime()

  dgemm_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  !write(*,'(a8,5i6)') 'ga_get:',mylo(1), myhi(1), mylo(2), myhi(2), myhi(2)-mylo(2)+1
  call ga_get( C, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )

  forder = real(order,REAL64)
  reference = 0.25d0 * forder**3 * (forder-1)**2 * (iterations+1)
  checksum = 0.0d0
  do j=mylo(2),myhi(2)
    jj = j-mylo(2)+1
    do i=mylo(1),myhi(1)
      ii = i-mylo(1)+1
      checksum = checksum + T(ii,jj)
    enddo
  enddo
  call ga_dgop(MT_DBL, checksum, 1, '+')

  deallocate( T )

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
    residuum = abs(checksum-reference)/reference
    if (residuum .lt. epsilon) then
      write(*,'(a)') 'Solution validates'
      avgtime = dgemm_time/iterations
      nflops = 2 * int(order,INT64)**3
      write(*,'(a,f13.3,a,f10.6)') 'Rate (MF/s): ',(1.d-6*nflops)/avgtime, &
             ' Avg time (s): ', avgtime
    else
      write(*,'(a,e30.15)') 'Reference checksum = ', reference
      write(*,'(a,e30.15)') 'Actual checksum    = ', checksum
      if (order.lt.10) then
        call ga_print(C)
      endif
      stop 1
    endif
  endif

  ok = ga_destroy(C)
  if (.not.ok) then
      call ga_error('ga_destroy failed',201)
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

