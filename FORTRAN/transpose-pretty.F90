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
!          transpose <matrix_size> <# iterations>
!
!          The output consists of diagnostics to make sure the
!          transpose worked and timing statistics.
!
! HISTORY: Written by  Rob Van der Wijngaart, February 2009.
!          Converted to Fortran by Jeff Hammond, February 2015
!
! *******************************************************************

program main
  use, intrinsic :: iso_fortran_env
  use prk
  implicit none
  integer :: err
  ! problem definition
  integer(kind=INT32) ::  iterations                ! number of times to do the transpose
  integer(kind=INT32) ::  order                     ! order of a the matrix
  integer(kind=INT32) ::  tile_size
  real(kind=REAL64), allocatable ::  A(:,:)         ! buffer to hold original matrix
  real(kind=REAL64), allocatable ::  B(:,:)         ! buffer to hold transposed matrix
  integer(kind=INT64) ::  bytes                     ! combined size of matrices
  ! runtime variables
  integer(kind=INT32) :: k
  integer(kind=INT64) :: j2, o2                      ! for loop over order**2
  real(kind=REAL64) ::  abserr                      ! squared error
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8    ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a40)') 'Fortran Pretty Matrix transpose: B = A^T'

  call prk_get_arguments('transpose',iterations=iterations,order=order,tile_size=tile_size)

  write(*,'(a22,i8)') 'Number of iterations = ', iterations
  write(*,'(a22,i8)') 'Matrix order         = ', order
  if (tile_size.ne.order) then
    write(*,'(a22,i8)') 'Tile size            = ', tile_size
  else
    write(*,'(a10)') 'Tiling off'
  endif

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(order,order), B(order,order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation  returned ',err
    stop 1
  endif

  t0 = 0

  ! Fill the original matrix
  o2 = int(order,INT64)**2
  A = reshape((/ (j2, j2 = 0,o2) /),(/order, order/))
  B = 0

  do k=0,iterations

    if (k.eq.1) t0 = prk_get_wtime()

    B = B + transpose(A)
    A = A + 1
  enddo ! iterations

  t1 = prk_get_wtime()
  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ! we reuse A here as the reference matrix, to compute the error
  A = ( transpose(reshape((/ (j2, j2 = 0,o2) /),(/order, order/))) &
        * real(iterations+1,REAL64) ) &
      + real((iterations*(iterations+1))/2,REAL64)
#if defined(PGI) || defined(XLF)
  abserr = 0.0d0
  do j=1,order
    do i=1,order
      abserr = abserr + (B(i,j) - A(i,j))**2
    enddo
  enddo
  abserr = sqrt(abserr)
#else
  abserr = norm2(A-B)
#endif

  deallocate( A,B )

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

