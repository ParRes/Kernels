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
!
! FUNCTIONS CALLED!:
!
!          Other than standard C functions, the following
!          functions are used in this program:
!
!          wtime()          portable wall-timer interface.
!
! HISTORY: Written by  Rob Van der Wijngaart, February 2009.
!          Converted to Fortran by Jeff Hammond, January 2015
! *******************************************************************

program main
  use iso_fortran_env
  implicit none
  ! for argument parsing
  integer :: err
  integer :: argnum, arglen
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

  print*,'Parallel Research Kernels version ', -1 !PRKVERSION
  print*,'Serial Matrix transpose: B = A^T'

  if (command_argument_count().lt.2) then
    print*,'argument count = ', command_argument_count()
    print*,'Usage: ./transpose <# iterations> <matrix order> [<tile_size>]'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i)') iterations

  order = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i)') order

  tile_size = order
  if (command_argument_count().gt.2) then
      call get_command_argument(3,argtmp,arglen,err)
      if (err.eq.0) read(argtmp,'(i)') tile_size
  endif

  if (iterations .lt. 1) then
    print*,'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  if (order .lt. 1) then
    print*,'ERROR: order must be >= 1 : ', order
    stop 1
  endif

  if ((tile_size .lt. 1).or.(tile_size.gt.order)) then
    print*,'WARNING: tile_size must be >= 1 and <= order : ', tile_size
    tile_size = order ! no tiling
  endif


  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(order,order), stat=err)
  if (err .ne. 0) then
    print*,'allocation of A returned ',err
    stop 1
  endif

  allocate( B(order,order), stat=err )
  if (err .ne. 0) then
    print*,'allocation of A returned ',err
    stop 1
  endif

  ! avoid overflow 64<-32
  bytes = 2 * order
  bytes = bytes * order

  print*,'Matrix order = ', order
  print*,'Number of iterations = ', iterations
  print*,'Tile size = ', tile_size

  ! Fill the original matrix, set transpose to known garbage value. */

  !  Fill the original column matrix
  do j=1,order
    do i=1,order
      A(i,j) = (i-1)+(j-1)*order
    enddo
  enddo

  !   Set the transpose matrix to a known garbage value.
  do j=1,order
    do i=1,order
      B(i,j) = 0.0
    enddo
  enddo

  do k=0,iterations

    !  start timer after a warmup iteration
    if (k.eq.1) call cpu_time(t0)

    !  Transpose the  matrix; only use tiling if the tile size is smaller than the matrix
    if (tile_size.lt.order) then
      do j=1,order,tile_size
        do i=1,order,tile_size
          do jt=j,min(order,j+tile_size-1)
            do it=i,min(order,i+tile_size-1)
              B(jt,it) = B(jt,it) + A(it,jt)
              A(it,jt) = A(it,jt) + 1.0
            enddo
          enddo
        enddo
      enddo
    else
      do j=1,order
        do i=1,order
          B(j,i) = B(j,i) + A(i,j)
          A(i,j) = A(i,j) + 1.0
        enddo
      enddo
    endif

  enddo ! iterations

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  call cpu_time(t1)
  trans_time = t1 - t0

  abserr = 0.0;
  addit = (0.5*iterations)
  addit = addit * (iterations+1)
  do j=1,order
    do i=1,order
      ! this was overflowing for iterations>15
      temp   = ((j-1.)+(i-1.)*order)
      temp   = temp * (iterations+1)
      abserr = abserr + abs(B(i,j) - (temp+addit))
    enddo
  enddo

  deallocate( B )
  deallocate( A )

  if (abserr .lt. epsilon) then
    print*,'Solution validates'
    avgtime = trans_time/iterations
    print*,'Rate (MB/s): ',1.e-6*bytes/avgtime, &
           ' Avg time (s): ', avgtime
    stop
  else
    print*,'ERROR: Aggregate squared error ',abserr, &
           'exceeds threshold ',epsilon
    stop 1
  endif

end program main

