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
! *******************************************************************

function prk_get_wtime() result(t)
  use iso_fortran_env
  implicit none
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
end function prk_get_wtime

program main
  use iso_fortran_env
  implicit none
  real(kind=REAL64) :: prk_get_wtime
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) ::  iterations                ! number of times to do the transpose
  integer(kind=INT32) ::  order                     ! order of a the matrix
  real(kind=REAL64), allocatable ::  A(:,:)         ! buffer to hold original matrix
  real(kind=REAL64), allocatable ::  B(:,:)         ! buffer to hold transposed matrix
  integer(kind=INT64) ::  bytes                     ! combined size of matrices
  ! runtime variables
  integer(kind=INT32) ::  i, j, k
  integer(kind=INT32) ::  it, jt, tile_size
  real(kind=REAL64) ::  abserr, addit, temp         ! squared error
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8    ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a41)') 'Fortran OpenACC Matrix transpose: B = A^T'

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a62)')    'Usage: ./transpose <# iterations> <matrix order> [<tile_size>]'
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

  ! same default as the C implementation
  tile_size = 32
  if (command_argument_count().gt.2) then
      call get_command_argument(3,argtmp,arglen,err)
      if (err.eq.0) read(argtmp,'(i32)') tile_size
  endif
  if ((tile_size .lt. 1).or.(tile_size.gt.order)) then
    write(*,'(a,i5,a,i5)') 'WARNING: tile_size ',tile_size,&
                           ' must be >= 1 and <= ',order
    tile_size = order ! no tiling
  endif

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(order,order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(order,order), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  write(*,'(a,i8)') 'Number of iterations = ', iterations
  write(*,'(a,i8)') 'Matrix order         = ', order
  write(*,'(a,i8)') 'Tile size            = ', tile_size

  t0 = 0

  if (tile_size.lt.order) then
    !$acc parallel loop gang ! collapse(2) leads to incorrect results
    do jt=1,order,tile_size
      !$acc loop
      do it=1,order,tile_size
        !$acc loop vector collapse(2)
        do j=jt,min(order,jt+tile_size-1)
          do i=it,min(order,it+tile_size-1)
              A(i,j) = real(order,REAL64) * real(j-1,REAL64) + real(i-1,REAL64)
              B(i,j) = 0.0
          enddo
        enddo
      enddo
    enddo
  else
    !$acc parallel loop gang
    do j=1,order
      !$acc loop vector
      do i=1,order
        A(i,j) = real(order,REAL64) * real(j-1,REAL64) + real(i-1,REAL64)
        B(i,j) = 0.0
      enddo
    enddo
  endif

  !$acc data pcopyin(A) pcopy(B)
  do k=0,iterations

    if (k.eq.1) t0 = prk_get_wtime()

    ! Transpose the matrix; only use tiling if the tile size is smaller than the matrix
    if (tile_size.lt.order) then
      !$acc parallel loop gang ! collapse(2) leads to incorrect results
      do jt=1,order,tile_size
        !$acc loop
        do it=1,order,tile_size
          !$acc loop vector collapse(2)
          do j=jt,min(order,jt+tile_size-1)
            do i=it,min(order,it+tile_size-1)
              B(j,i) = B(j,i) + A(i,j)
              A(i,j) = A(i,j) + 1.0
            enddo
          enddo
        enddo
      enddo
    else
      !$acc parallel loop gang
      do j=1,order
        !$acc loop vector
        do i=1,order
          B(j,i) = B(j,i) + A(i,j)
          A(i,j) = A(i,j) + 1.0
        enddo
      enddo
    endif

  enddo ! iterations

  t1 = prk_get_wtime()

  !$acc end data

  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  abserr = 0.0
  ! this will overflow if iterations>>1000
  addit = (0.5*iterations) * (iterations+1)
  !$acc parallel loop reduction(+:abserr)
  do j=1,order
    !$acc loop reduction(+:abserr)
    do i=1,order
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(j-1,REAL64)) &
           * real(iterations+1,REAL64)
      abserr = abserr + abs(B(i,j) - (temp+addit))
    enddo
  enddo

  deallocate( B )
  deallocate( A )

  if (abserr .lt. epsilon) then
    write(*,'(a)') 'Solution validates'
    avgtime = trans_time/iterations
    bytes = 2 * int(order,INT64) * int(order,INT64) * storage_size(A)/8
    write(*,'(a,f13.6,a,f10.6)') 'Rate (MB/s): ',(1.d-6*bytes)/avgtime, &
           ' Avg time (s): ', avgtime
  else
    write(*,'(a,f30.15,a,f30.15)') 'ERROR: Aggregate squared error ',abserr, &
           'exceeds threshold ',epsilon
    stop 1
  endif

end program main

