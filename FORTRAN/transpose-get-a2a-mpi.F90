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

module prk_mpi
  contains
    subroutine mpi_print_matrix(mat,clabel)
      use iso_fortran_env
      use mpi_f08
      use prk
      implicit none
      real(kind=REAL64), intent(in) :: mat(:,:)
      character(*), intent(in), optional :: clabel
      integer(kind=INT32) :: r, me, np
      flush(6)
      call MPI_Comm_rank(MPI_COMM_WORLD, me)
      call MPI_Comm_size(MPI_COMM_WORLD, np)
      call MPI_Barrier(MPI_COMM_WORLD)
      flush(6)
      if (me.eq.0) print*,clabel
      flush(6)
      call MPI_Barrier(MPI_COMM_WORLD)
      flush(6)
      do r=0,np-1
        if (me.eq.r) then
          call print_matrix(mat,me)
        endif
        call MPI_Barrier(MPI_COMM_WORLD)
      enddo
      flush(6)
    end subroutine
end module prk_mpi

program main
  use iso_fortran_env
  use mpi_f08
  use prk
  use prk_mpi
  implicit none
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) ::  order, block_order
  type(MPI_Win) :: WA                               ! MPI window for A (original matrix)
  type(c_ptr) :: XA                                 ! MPI baseptr / C pointer for A
  real(kind=REAL64), pointer     ::  A(:,:)         ! Fortran pointer to A
  real(kind=REAL64), allocatable ::  B(:,:)         ! buffer to hold transposed matrix
  real(kind=REAL64), allocatable ::  T(:,:)         ! temporary to hold tile
  real(kind=REAL64), parameter :: one=1.0d0
  ! runtime variables
  integer(kind=INT64) ::  bytes
  integer(kind=INT32) ::  i, j, k, r, lo, hi
  !integer(kind=INT32) ::  it, jt, tile_size
  real(kind=REAL64) ::  abserr, addit, temp
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.d-8
  ! MPI stuff
  integer(kind=INT32) :: me, np, provided
  integer(kind=MPI_ADDRESS_KIND) :: wsize, woff
  integer(kind=INT32) :: dsize

  call MPI_Init_thread(MPI_THREAD_SINGLE,provided)
  call MPI_Comm_rank(MPI_COMM_WORLD, me)
  call MPI_Comm_size(MPI_COMM_WORLD, np)

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (me.eq.0) then
    write(*,'(a25)') 'Parallel Research Kernels'
    write(*,'(a36)') 'Fortran MPI Matrix transpose: B = A^T'

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

    write(*,'(a23,i8)') 'Number of MPI procs  = ', np
    write(*,'(a23,i8)') 'Number of iterations = ', iterations
    write(*,'(a23,i8)') 'Matrix order         = ', order
  endif
  call MPI_Bcast(iterations, 1, MPI_INTEGER4, 0, MPI_COMM_WORLD)
  call MPI_Bcast(order, 1, MPI_INTEGER4, 0, MPI_COMM_WORLD)

  block_order = int(order / np)

  call MPI_Barrier(MPI_COMM_WORLD)

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  dsize = storage_size(one)/8
  ! MPI_Win_allocate(size, disp_unit, info, comm, baseptr, win, ierror)
  wsize = block_order * order * dsize
  call MPI_Win_allocate(size=wsize, disp_unit=dsize, &
                        info=MPI_INFO_NULL, comm=MPI_COMM_WORLD, baseptr=XA, win=WA)
  call MPI_Win_lock_all(0,WA)

  call c_f_pointer(XA,A,[block_order,order])
                        
  allocate( B(block_order,order), T(block_order,order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation  returned ',err
    stop 1
  endif
  
  ! Fill the original matrix
  do concurrent (i=1:order, j=1:block_order)
    A(j,i) = me * block_order + (i-1)*order + (j-1)
  end do
  call MPI_Win_sync(WA)
  B = 0

  t0 = 0.0d0

  do k=0,iterations

    if (k.eq.1) then
        call MPI_Barrier(MPI_COMM_WORLD)
        t0 = MPI_Wtime()
    endif

    ! B += A^T
    call MPI_Barrier(MPI_COMM_WORLD)
    do r=0,np-1
        woff = block_order * block_order * me
        lo = block_order * r + 1
        hi = block_order * (r+1)
        call MPI_Get(origin_addr=T(:,lo:hi), origin_count=block_order*block_order, &
                     origin_datatype=MPI_DOUBLE_PRECISION, &
                     target_rank=r, target_disp=woff, target_count=block_order*block_order, &
                     target_datatype=MPI_DOUBLE_PRECISION, win=WA)
    end do
    call MPI_Win_flush_local_all(WA)
    do r=0,np-1
        lo = block_order * r + 1
        hi = block_order * (r+1)
        B(:,lo:hi) = B(:,lo:hi) + transpose(T(:,lo:hi))
    end do
    ! A += 1
    A = A + one
    call MPI_Win_sync(WA)

  enddo ! iterations

  call MPI_Barrier(MPI_COMM_WORLD)
  t1 = MPI_Wtime()

  trans_time = t1 - t0

  deallocate( T )
  call MPI_Win_unlock_all(WA)
  call MPI_Win_free( WA)

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

